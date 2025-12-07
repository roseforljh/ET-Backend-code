import logging
import base64
import asyncio
import orjson
from typing import AsyncGenerator, Dict, Any
from .voice import google_handler, openai_handler, minimax_handler, siliconflow_handler, aliyun_handler
from .voice.validator import validate_voice_config
from .smart_splitter import SmartSentenceSplitter
from .predictive_tts import PredictiveTTSManager
from ..utils.helpers import strip_markdown_for_tts

logger = logging.getLogger("EzTalkProxy.Services.VoiceStreaming")

# 预测性 TTS 配置
PREDICTIVE_TTS_CONFIG = {
    "max_concurrent": 5,       # 最大并发 TTS 请求
    "max_retry": 2,            # 最大重试次数
    "task_timeout": 30.0,      # 单任务超时（秒）
}

# 音频格式配置
# 不同 TTS 平台的音频格式
# 注意：SiliconFlow 的 Opus 格式与 Android MediaCodec 解码器存在兼容性问题
# 临时使用 PCM 格式以确保稳定性
AUDIO_FORMAT_CONFIG = {
    "Gemini": {"format": "pcm", "sample_rate": 24000},        # Gemini SDK 仅返回 PCM
    "Minimax": {"format": "pcm", "sample_rate": 24000},       # 流式需要 PCM（MP3 需完整帧）
    "SiliconFlow": {"format": "pcm", "sample_rate": 32000},   # PCM 格式，32kHz（避免 Opus 解码问题）
    "OpenAI": {"format": "opus", "sample_rate": 24000},       # ✅ Opus 压缩
}


class VoiceStreamProcessor:
    def __init__(
        self,
        stt_config: Dict[str, Any],
        chat_config: Dict[str, Any],
        tts_config: Dict[str, Any],
        use_predictive_tts: bool = True,  # 是否使用预测性 TTS
    ):
        self.stt_config = stt_config
        self.chat_config = chat_config
        self.tts_config = tts_config
        self.use_predictive_tts = use_predictive_tts
        self.sentence_buffer = ""
        self.full_assistant_text = ""
        
        # 使用智能分割器替代简单的正则匹配
        # 配置：最小8字符，理想20字符，最大50字符，绝对最大80字符
        self.splitter = SmartSentenceSplitter(
            min_length=8,
            preferred_length=20,
            max_length=50,
            absolute_max=80
        )

    async def process(self, audio_bytes: bytes, mime_type: str) -> AsyncGenerator[bytes, None]:
        """
        Execute the full STT -> Chat Stream -> TTS Stream pipeline.
        Yields NDJSON bytes.
        
        支持两种模式：
        1. 预测性 TTS（默认）：LLM 和 TTS 并行执行，减少延迟
        2. Stop-and-Go：传统模式，每个片段等待 TTS 完成后继续
        """
        # 1. STT
        user_text = await self._run_stt(audio_bytes, mime_type)
        if not user_text:
            yield self._format_event("error", {"message": "STT failed or empty result"})
            return

        # 获取音频格式配置
        tts_platform = self.tts_config.get("platform", "Gemini")
        audio_config = AUDIO_FORMAT_CONFIG.get(tts_platform, AUDIO_FORMAT_CONFIG["Gemini"])
        
        # Yield initial meta event with user text and audio format
        yield self._format_event("meta", {
            "user_text": user_text,
            "assistant_text": "",
            "audio_format": audio_config["format"],
            "sample_rate": audio_config["sample_rate"]
        })

        # 根据配置选择处理模式
        if self.use_predictive_tts:
            async for event in self._process_with_predictive_tts(user_text):
                yield event
        else:
            async for event in self._process_with_stop_and_go(user_text):
                yield event

    async def _process_with_predictive_tts(self, user_text: str) -> AsyncGenerator[bytes, None]:
        """
        预测性 TTS 处理流程
        
        LLM 输出和 TTS 合成并行执行，显著减少响应延迟。
        """
        # 创建预测性 TTS 管理器和处理器
        tts_manager = PredictiveTTSManager(self.tts_config)
        tts_processor = tts_manager.create_processor(
            max_concurrent=PREDICTIVE_TTS_CONFIG["max_concurrent"],
            max_retry=PREDICTIVE_TTS_CONFIG["max_retry"],
            task_timeout=PREDICTIVE_TTS_CONFIG["task_timeout"]
        )
        
        sequence_id = 0
        
        # 用于存储需要输出的事件
        meta_events = []
        
        try:
            # 2. Chat Stream + 提交 TTS 任务
            async for token in self._run_chat_stream(user_text):
                self.sentence_buffer += token
                self.full_assistant_text += token

                # 使用智能分割器分割文本
                result = self.splitter.split(self.sentence_buffer)
                
                # 处理每个可发送的片段
                for segment in result.segments:
                    if not segment.strip():
                        continue
                    
                    # 发送 meta 更新
                    yield self._format_event("meta", {
                        "user_text": user_text,
                        "assistant_text": self.full_assistant_text
                    })
                    
                    # 提交 TTS 任务（非阻塞）
                    clean_segment = strip_markdown_for_tts(segment)
                    await tts_processor.submit_task(sequence_id, clean_segment)
                    sequence_id += 1
                
                # 更新 buffer 为剩余内容
                self.sentence_buffer = result.remainder

            # 处理剩余 buffer
            if self.sentence_buffer.strip():
                yield self._format_event("meta", {
                    "user_text": user_text,
                    "assistant_text": self.full_assistant_text
                })
                clean_segment = strip_markdown_for_tts(self.sentence_buffer)
                await tts_processor.submit_task(sequence_id, clean_segment)
                sequence_id += 1
            
            # 标记输入完成
            tts_processor.mark_input_complete()
            
            # 3. 按顺序输出音频
            async for audio_chunk in tts_processor.yield_audio_in_order():
                if audio_chunk:
                    yield self._format_event("audio", {
                        "data": base64.b64encode(audio_chunk).decode('utf-8')
                    })
                    
        except Exception as e:
            logger.exception("Predictive TTS pipeline error")
            yield self._format_event("error", {"message": str(e)})
        finally:
            # 清理资源
            await tts_processor.cleanup()

    async def _process_with_stop_and_go(self, user_text: str) -> AsyncGenerator[bytes, None]:
        """
        传统 Stop-and-Go 处理流程
        
        每个片段等待 TTS 完成后再处理下一个。
        """
        try:
            async for token in self._run_chat_stream(user_text):
                self.sentence_buffer += token
                self.full_assistant_text += token

                # 使用智能分割器分割文本
                result = self.splitter.split(self.sentence_buffer)
                
                # 处理每个可发送的片段
                for segment in result.segments:
                    if not segment.strip():
                        continue
                        
                    # Yield updated text
                    yield self._format_event("meta", {
                        "user_text": user_text,
                        "assistant_text": self.full_assistant_text
                    })
                    
                    # TTS Stream (Stop-and-Go)
                    clean_segment = strip_markdown_for_tts(segment)
                    try:
                        async for audio_chunk in self._run_tts_stream(clean_segment):
                            if audio_chunk:
                                yield self._format_event("audio", {
                                    "data": base64.b64encode(audio_chunk).decode('utf-8')
                                })
                    except Exception as tts_error:
                        yield self._format_event("error", {"message": f"TTS Error: {str(tts_error)}"})
                
                # 更新 buffer 为剩余内容
                self.sentence_buffer = result.remainder

            # Flush remaining buffer
            if self.sentence_buffer.strip():
                yield self._format_event("meta", {
                    "user_text": user_text,
                    "assistant_text": self.full_assistant_text
                })
                clean_segment = strip_markdown_for_tts(self.sentence_buffer)
                try:
                    async for audio_chunk in self._run_tts_stream(clean_segment):
                        if audio_chunk:
                            yield self._format_event("audio", {
                                "data": base64.b64encode(audio_chunk).decode('utf-8')
                            })
                except Exception as tts_error:
                    yield self._format_event("error", {"message": f"TTS Error: {str(tts_error)}"})

        except Exception as e:
            logger.exception("Streaming pipeline error")
            yield self._format_event("error", {"message": str(e)})

    async def _run_stt(self, audio_bytes: bytes, mime_type: str) -> str:
        platform = self.stt_config.get("platform", "Google")
        try:
            if platform == "OpenAI":
                return await openai_handler.process_stt(
                    audio_bytes=audio_bytes,
                    api_key=self.stt_config["api_key"],
                    api_url=self.stt_config.get("api_url"),
                    model=self.stt_config["model"]
                )
            elif platform == "SiliconFlow":
                return await siliconflow_handler.process_stt(
                    audio_bytes=audio_bytes,
                    api_key=self.stt_config["api_key"],
                    api_url=self.stt_config["api_url"],
                    model=self.stt_config["model"],
                    mime_type=mime_type
                )
            elif platform == "Aliyun":
                fmt = "opus" if "opus" in mime_type or "ogg" in mime_type else "wav"
                return await aliyun_handler.process_stt(
                    audio_bytes=audio_bytes,
                    api_key=self.stt_config["api_key"],
                    api_url=self.stt_config.get("api_url"),
                    model=self.stt_config["model"],
                    format=fmt,
                    sample_rate=16000
                )
            else: # Google
                return google_handler.process_stt(
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                    api_key=self.stt_config["api_key"],
                    model=self.stt_config["model"],
                    api_url=self.stt_config.get("api_url")
                )
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return ""

    async def _run_chat_stream(self, user_text: str):
        platform = self.chat_config.get("platform", "Google")
        
        # Construct system prompt
        system_prompt = self.chat_config.get("system_prompt", "")
        
        # SiliconFlow and other OpenAI-compatible platforms
        if platform == "OpenAI" or platform == "SiliconFlow":
            async for chunk in openai_handler.process_chat_stream(
                user_text=user_text,
                chat_history=self.chat_config.get("history", []),
                system_prompt=system_prompt,
                api_key=self.chat_config["api_key"],
                api_url=self.chat_config.get("api_url"),
                model=self.chat_config["model"]
            ):
                yield chunk
        else: # Google
            for chunk in google_handler.process_chat_stream(
                user_text=user_text,
                chat_history=self.chat_config.get("history", []),
                system_prompt=system_prompt,
                api_key=self.chat_config["api_key"],
                model=self.chat_config["model"],
                api_url=self.chat_config.get("api_url")
            ):
                yield chunk

    async def _run_tts_stream(self, text: str):
        platform = self.tts_config.get("platform", "Gemini")
        voice_name = self.tts_config.get("voice_name", "Kore")
        
        # 使用统一校验器
        validate_voice_config(platform, voice_name)

        try:
            if platform == "Minimax":
                async for chunk in minimax_handler.synthesize_minimax_t2a_stream(
                    text=text,
                    voice_id=voice_name,
                    api_url=self.tts_config["api_url"],
                    api_key=self.tts_config["api_key"]
                ):
                    yield chunk
            elif platform == "SiliconFlow":
                # 使用配置的音频格式
                audio_config = AUDIO_FORMAT_CONFIG.get("SiliconFlow", AUDIO_FORMAT_CONFIG["Gemini"])
                async for chunk in siliconflow_handler.process_tts_stream(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    api_url=self.tts_config["api_url"],
                    model=self.tts_config["model"],
                    voice=voice_name,
                    response_format=audio_config["format"],
                    sample_rate=audio_config["sample_rate"]
                ):
                    yield chunk
            elif platform == "OpenAI":
                # OpenAI TTS is not truly streaming (returns full file), but we can chunk it
                wav_data = await openai_handler.process_tts(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    voice_name=voice_name,
                    model=self.tts_config["model"],
                    api_url=self.tts_config.get("api_url")
                )
                if not wav_data:
                     raise ValueError("OpenAI TTS returned empty data")
                yield wav_data
            else: # Gemini
                pcm_data = google_handler.process_tts(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    voice_name=voice_name,
                    model=self.tts_config["model"],
                    api_url=self.tts_config.get("api_url")
                )
                if not pcm_data:
                     raise ValueError("Gemini TTS returned empty data")
                yield pcm_data
        except Exception as e:
            logger.error(f"TTS stream failed for chunk '{text[:20]}...': {e}")
            # 重新抛出异常，以便上层捕获并发送 error 事件给前端
            raise e

    def _format_event(self, event_type: str, data: Dict[str, Any]) -> bytes:
        payload = {"type": event_type, **data}
        return orjson.dumps(payload) + b"\n"
