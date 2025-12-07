"""
实时语音对话 WebSocket API

端到端流式处理：
1. 客户端边录音边发送音频块
2. 服务端实时进行 STT 识别
3. STT 完成后流式调用 Chat
4. Chat 输出流式发送给 TTS
5. TTS 音频实时返回客户端

协议说明见架构文档
"""

import logging
import base64
import asyncio
import orjson
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.voice.aliyun_streaming import AliyunRealtimeSTT, STTConfig, STTState
from ..services.voice import google_handler, openai_handler, minimax_handler, siliconflow_handler
from ..services.voice.validator import validate_voice_config
from ..services.smart_splitter import SmartSentenceSplitter
from ..services.predictive_tts import PredictiveTTSManager
from ..utils.helpers import strip_markdown_for_tts
from ..services.requests.prompts import compose_voice_system_prompt

logger = logging.getLogger("EzTalkProxy.API.VoiceChatWS")
router = APIRouter()

# 音频格式配置
AUDIO_FORMAT_CONFIG = {
    "Gemini": {"format": "pcm", "sample_rate": 24000},
    "Minimax": {"format": "pcm", "sample_rate": 24000},
    "SiliconFlow": {"format": "pcm", "sample_rate": 32000},
    "OpenAI": {"format": "opus", "sample_rate": 24000},
}

# 预测性 TTS 配置
PREDICTIVE_TTS_CONFIG = {
    "max_concurrent": 5,
    "max_retry": 2,
    "task_timeout": 30.0,
}


class RealtimeVoiceChatSession:
    """
    实时语音对话会话
    
    管理单个 WebSocket 连接的完整对话流程
    """
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stt: Optional[AliyunRealtimeSTT] = None
        self.stt_config: Optional[Dict[str, Any]] = None
        self.chat_config: Optional[Dict[str, Any]] = None
        self.tts_config: Optional[Dict[str, Any]] = None
        self.is_cancelled = False
        self.splitter = SmartSentenceSplitter(
            min_length=8,
            preferred_length=20,
            max_length=50,
            absolute_max=80
        )
    
    async def send_json(self, data: Dict[str, Any]):
        """发送 JSON 消息"""
        try:
            await self.websocket.send_bytes(orjson.dumps(data))
        except Exception as e:
            logger.warning(f"发送消息失败: {e}")
    
    async def send_error(self, message: str):
        """发送错误消息"""
        await self.send_json({"type": "error", "message": message})
    
    async def handle_init(self, msg: Dict[str, Any]) -> bool:
        """处理初始化消息"""
        try:
            self.stt_config = msg.get("stt", {})
            self.chat_config = msg.get("chat", {})
            self.tts_config = msg.get("tts", {})
            
            # 验证必需参数
            if not self.stt_config.get("api_key"):
                await self.send_error("STT API Key 未提供")
                return False
            
            if not self.chat_config.get("api_key"):
                await self.send_error("Chat API Key 未提供")
                return False
            
            # 仅阿里云支持真正的流式 STT
            stt_platform = self.stt_config.get("platform", "Aliyun")
            if stt_platform != "Aliyun":
                await self.send_error(f"实时流式 STT 仅支持 Aliyun 平台，当前: {stt_platform}")
                return False
            
            # 创建 STT 实例
            config = STTConfig(
                api_key=self.stt_config["api_key"],
                model=self.stt_config.get("model", "fun-asr-realtime"),
                sample_rate=self.stt_config.get("sample_rate", 16000),
                format=self.stt_config.get("format", "pcm"),
                vocabulary_id=self.stt_config.get("vocabulary_id")
            )
            
            self.stt = AliyunRealtimeSTT(config)
            
            # 启动 STT
            success = await self.stt.start()
            if not success:
                await self.send_error("STT 启动失败")
                return False
            
            await self.send_json({"type": "ready"})
            logger.info("实时语音对话会话已初始化")
            return True
            
        except Exception as e:
            logger.exception("初始化失败")
            await self.send_error(f"初始化失败: {str(e)}")
            return False
    
    async def handle_audio(self, msg: Dict[str, Any]):
        """处理音频数据"""
        if not self.stt or self.stt.get_state() != STTState.STREAMING:
            return
        
        try:
            # 解码 Base64 音频数据
            audio_data = base64.b64decode(msg.get("data", ""))
            if audio_data:
                await self.stt.send_audio(audio_data)
            
            # 检查是否有识别结果
            result = await self.stt.get_result(timeout=0.01)
            if result:
                await self._handle_stt_result(result)
                
        except Exception as e:
            logger.error(f"处理音频失败: {e}")
    
    async def _handle_stt_result(self, result: Dict[str, Any]):
        """处理 STT 结果"""
        result_type = result.get("type")
        
        if result_type == "partial":
            await self.send_json({
                "type": "stt_partial",
                "text": result.get("text", ""),
                "is_final": False
            })
        elif result_type == "sentence":
            await self.send_json({
                "type": "stt_partial",
                "text": result.get("text", ""),
                "is_final": True,
                "accumulated": result.get("accumulated", "")
            })
        elif result_type == "error":
            await self.send_error(f"STT 错误: {result.get('message')}")
    
    async def handle_end(self) -> Optional[str]:
        """
        处理结束信号
        
        Returns:
            最终识别文本
        """
        if not self.stt:
            return None
        
        try:
            # 获取所有剩余的 STT 结果
            while True:
                result = await self.stt.get_result(timeout=0.1)
                if result:
                    await self._handle_stt_result(result)
                else:
                    break
            
            # 停止 STT 并获取最终结果
            final_text = await self.stt.stop()
            
            await self.send_json({
                "type": "stt_final",
                "text": final_text
            })
            
            return final_text
            
        except Exception as e:
            logger.error(f"结束 STT 失败: {e}")
            return None
    
    async def run_chat_and_tts(self, user_text: str):
        """
        运行 Chat 和 TTS 流程
        
        Args:
            user_text: 用户输入文本（STT 结果）
        """
        if not user_text or self.is_cancelled:
            return
        
        # 获取音频格式配置
        tts_platform = self.tts_config.get("platform", "Gemini")
        audio_config = AUDIO_FORMAT_CONFIG.get(tts_platform, AUDIO_FORMAT_CONFIG["Gemini"])
        
        # 发送音频格式信息
        await self.send_json({
            "type": "meta",
            "audio_format": audio_config["format"],
            "sample_rate": audio_config["sample_rate"]
        })
        
        # 构建系统提示
        voice_prompt = compose_voice_system_prompt()
        system_prompt = self.chat_config.get("system_prompt", "")
        if system_prompt:
            final_system_prompt = f"{voice_prompt}\n\n[补充要求]\n{system_prompt}"
        else:
            final_system_prompt = voice_prompt
        
        # 创建 TTS 管理器
        tts_manager = PredictiveTTSManager(self.tts_config)
        tts_processor = tts_manager.create_processor(
            max_concurrent=PREDICTIVE_TTS_CONFIG["max_concurrent"],
            max_retry=PREDICTIVE_TTS_CONFIG["max_retry"],
            task_timeout=PREDICTIVE_TTS_CONFIG["task_timeout"]
        )
        
        sentence_buffer = ""
        full_text = ""
        sequence_id = 0
        
        try:
            # 流式 Chat
            async for token in self._run_chat_stream(user_text, final_system_prompt):
                if self.is_cancelled:
                    break
                
                sentence_buffer += token
                full_text += token
                
                # 发送增量文本
                await self.send_json({
                    "type": "chat_delta",
                    "text": token,
                    "full_text": full_text
                })
                
                # 使用智能分割器
                result = self.splitter.split(sentence_buffer)
                
                for segment in result.segments:
                    if not segment.strip():
                        continue
                    
                    # 提交 TTS 任务
                    clean_segment = strip_markdown_for_tts(segment)
                    await tts_processor.submit_task(sequence_id, clean_segment)
                    sequence_id += 1
                
                sentence_buffer = result.remainder
            
            # 处理剩余 buffer
            if sentence_buffer.strip() and not self.is_cancelled:
                clean_segment = strip_markdown_for_tts(sentence_buffer)
                await tts_processor.submit_task(sequence_id, clean_segment)
                sequence_id += 1
            
            # 标记输入完成
            tts_processor.mark_input_complete()
            
            # 按顺序输出音频
            async for audio_chunk in tts_processor.yield_audio_in_order():
                if self.is_cancelled:
                    break
                if audio_chunk:
                    await self.send_json({
                        "type": "audio",
                        "data": base64.b64encode(audio_chunk).decode('utf-8')
                    })
            
            # 发送完成信号
            await self.send_json({
                "type": "complete",
                "user_text": user_text,
                "assistant_text": full_text
            })
            
        except Exception as e:
            logger.exception("Chat/TTS 流程错误")
            await self.send_error(f"处理失败: {str(e)}")
        finally:
            await tts_processor.cleanup()
    
    async def _run_chat_stream(self, user_text: str, system_prompt: str):
        """运行流式 Chat"""
        platform = self.chat_config.get("platform", "Google")
        history = self.chat_config.get("history", [])
        
        if platform in ("OpenAI", "SiliconFlow"):
            async for chunk in openai_handler.process_chat_stream(
                user_text=user_text,
                chat_history=history,
                system_prompt=system_prompt,
                api_key=self.chat_config["api_key"],
                api_url=self.chat_config.get("api_url"),
                model=self.chat_config.get("model", "gpt-4")
            ):
                yield chunk
        else:  # Google
            for chunk in google_handler.process_chat_stream(
                user_text=user_text,
                chat_history=history,
                system_prompt=system_prompt,
                api_key=self.chat_config["api_key"],
                model=self.chat_config.get("model", "gemini-pro"),
                api_url=self.chat_config.get("api_url")
            ):
                yield chunk
    
    async def cancel(self):
        """取消会话"""
        self.is_cancelled = True
        if self.stt:
            await self.stt.cancel()


@router.websocket("/realtime")
async def voice_chat_ws(websocket: WebSocket):
    """
    实时语音对话 WebSocket 端点
    
    协议：
    - 客户端发送 init 消息进行初始化
    - 客户端发送 audio 消息传输音频块
    - 客户端发送 end 消息结束录音
    - 客户端发送 cancel 消息取消会话
    - 服务端发送各种状态和数据消息
    """
    await websocket.accept()
    
    session = RealtimeVoiceChatSession(websocket)
    logger.info(f"WebSocket 连接请求: {websocket.client}")
    
    try:
        # 启动心跳任务
        async def heartbeat():
            while True:
                try:
                    await asyncio.sleep(15)
                    await session.send_json({"type": "ping"})
                except Exception:
                    break
        
        heartbeat_task = asyncio.create_task(heartbeat())
        
        try:
            while True:
                # 使用通用 receive() 方法统一处理文本和二进制消息
                # 这样可以避免消息类型不匹配导致的消息丢失问题
                try:
                    message = await websocket.receive()
                except Exception as e:
                    logger.warning(f"接收消息失败: {e}")
                    break
                
                # 检查连接是否断开
                if message["type"] == "websocket.disconnect":
                    logger.info("收到断开连接消息")
                    break
                
                # 解析消息内容（支持文本和二进制格式）
                try:
                    if "bytes" in message and message["bytes"]:
                        msg = orjson.loads(message["bytes"])
                    elif "text" in message and message["text"]:
                        msg = orjson.loads(message["text"])
                    else:
                        logger.warning(f"收到空消息或未知格式: {message.get('type')}")
                        continue
                except orjson.JSONDecodeError as e:
                    logger.warning(f"JSON 解析失败: {e}")
                    continue
                
                msg_type = msg.get("type")
                logger.info(f"收到消息: {msg_type}")
                
                if msg_type == "init":
                    success = await session.handle_init(msg)
                    if not success:
                        break
                
                elif msg_type == "audio":
                    await session.handle_audio(msg)
                
                elif msg_type == "end":
                    # 结束录音，获取最终 STT 结果
                    final_text = await session.handle_end()
                    
                    if final_text:
                        # 运行 Chat 和 TTS
                        await session.run_chat_and_tts(final_text)
                    
                    break
                
                elif msg_type == "cancel":
                    await session.cancel()
                    break
                
                elif msg_type == "pong":
                    # 心跳响应
                    pass
                
                else:
                    logger.warning(f"未知消息类型: {msg_type}")
        
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
    except Exception as e:
        logger.exception("WebSocket 处理错误")
    finally:
        await session.cancel()
        try:
            await websocket.close()
        except Exception:
            pass