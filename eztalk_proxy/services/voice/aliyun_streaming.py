"""
阿里云 Fun-ASR 实时语音识别流式适配器

基于 DashScope SDK 实现真正的流式语音识别：
- 边录音边发送音频块
- 实时返回识别结果
- 支持 VAD 断句和语义断句

参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/funasr-real-time-speech-recognition-api
"""

import logging
import asyncio
import threading
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("EzTalkProxy.Services.Voice.AliyunStreaming")

# 尝试导入 DashScope SDK
try:
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
    _DASHSCOPE_AVAILABLE = True
except ImportError:
    _DASHSCOPE_AVAILABLE = False
    logger.warning("DashScope SDK 不可用，请安装: pip install dashscope>=1.23.1")


class STTState(Enum):
    """STT 状态机"""
    IDLE = "idle"
    STARTING = "starting"
    STREAMING = "streaming"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class STTConfig:
    """STT 配置"""
    api_key: str
    model: str = "fun-asr-realtime"
    sample_rate: int = 16000
    format: str = "pcm"
    # VAD 断句配置（低延迟，适合交互场景）
    semantic_punctuation_enabled: bool = False
    max_sentence_silence: int = 800  # 降低静音阈值以加快断句
    # 热词
    vocabulary_id: Optional[str] = None


class AliyunRealtimeSTTCallback(RecognitionCallback):
    """
    阿里云实时 STT 回调处理器
    
    将 SDK 的同步回调转换为异步队列消息
    """
    
    def __init__(
        self,
        result_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop
    ):
        self.result_queue = result_queue
        self.loop = loop
        self._final_text = ""
    
    def on_open(self):
        """连接建立"""
        logger.info("Aliyun STT: WebSocket 连接已建立")
        asyncio.run_coroutine_threadsafe(
            self.result_queue.put({"type": "open"}),
            self.loop
        )
    
    def on_close(self):
        """连接关闭"""
        logger.info("Aliyun STT: WebSocket 连接已关闭")
        asyncio.run_coroutine_threadsafe(
            self.result_queue.put({"type": "close"}),
            self.loop
        )
    
    def on_event(self, result: RecognitionResult):
        """识别事件（实时结果）"""
        try:
            sentence = result.get_sentence()
            if sentence:
                # 提取文本
                if isinstance(sentence, dict):
                    text = sentence.get('text', '')
                    is_final = RecognitionResult.is_sentence_end(sentence)
                elif isinstance(sentence, list) and len(sentence) > 0:
                    # 多句结果，取最后一句
                    last_sentence = sentence[-1]
                    text = last_sentence.get('text', '') if isinstance(last_sentence, dict) else ''
                    is_final = RecognitionResult.is_sentence_end(last_sentence) if isinstance(last_sentence, dict) else False
                else:
                    text = ''
                    is_final = False
                
                if text:
                    if is_final:
                        self._final_text += text + " "
                        logger.debug(f"STT 句子完成: {text}")
                    else:
                        logger.debug(f"STT 部分结果: {text}")
                    
                    asyncio.run_coroutine_threadsafe(
                        self.result_queue.put({
                            "type": "partial" if not is_final else "sentence",
                            "text": text,
                            "is_final": is_final,
                            "accumulated": self._final_text.strip()
                        }),
                        self.loop
                    )
        except Exception as e:
            logger.error(f"处理 STT 事件失败: {e}")
    
    def on_complete(self):
        """识别完成"""
        logger.info(f"Aliyun STT: 识别完成，最终文本: {self._final_text.strip()}")
        asyncio.run_coroutine_threadsafe(
            self.result_queue.put({
                "type": "complete",
                "text": self._final_text.strip()
            }),
            self.loop
        )
    
    def on_error(self, result: RecognitionResult):
        """识别错误"""
        error_msg = str(result) if result else "Unknown error"
        logger.error(f"Aliyun STT 错误: {error_msg}")
        asyncio.run_coroutine_threadsafe(
            self.result_queue.put({
                "type": "error",
                "message": error_msg
            }),
            self.loop
        )


class AliyunRealtimeSTT:
    """
    阿里云实时语音识别适配器
    
    使用方法:
    ```python
    stt = AliyunRealtimeSTT(config)
    await stt.start()
    
    # 循环发送音频块
    for chunk in audio_chunks:
        await stt.send_audio(chunk)
        
        # 获取实时结果（非阻塞）
        result = await stt.get_result(timeout=0.01)
        if result:
            print(result)
    
    # 结束识别
    final_text = await stt.stop()
    ```
    """
    
    def __init__(self, config: STTConfig):
        if not _DASHSCOPE_AVAILABLE:
            raise RuntimeError("DashScope SDK 不可用，请安装: pip install dashscope>=1.23.1")
        
        self.config = config
        self.state = STTState.IDLE
        self.recognition: Optional[Recognition] = None
        self.callback: Optional[AliyunRealtimeSTTCallback] = None
        self.result_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._final_text = ""
    
    async def start(self) -> bool:
        """启动流式识别"""
        if self.state != STTState.IDLE:
            logger.warning(f"无法启动 STT，当前状态: {self.state}")
            return False
        
        self.state = STTState.STARTING
        self._loop = asyncio.get_event_loop()
        self.result_queue = asyncio.Queue()
        
        try:
            # 创建回调处理器
            self.callback = AliyunRealtimeSTTCallback(
                result_queue=self.result_queue,
                loop=self._loop
            )
            
            # 创建识别对象
            recognition_params = {
                "model": self.config.model,
                "format": self.config.format,
                "sample_rate": self.config.sample_rate,
                "api_key": self.config.api_key,
                "callback": self.callback,
                "semantic_punctuation_enabled": self.config.semantic_punctuation_enabled,
                "max_sentence_silence": self.config.max_sentence_silence,
            }
            
            if self.config.vocabulary_id:
                recognition_params["vocabulary_id"] = self.config.vocabulary_id
            
            self.recognition = Recognition(**recognition_params)
            
            # 启动流式识别（在后台线程中执行，因为 SDK 可能是阻塞的）
            def start_recognition():
                try:
                    self.recognition.start()
                except Exception as e:
                    logger.error(f"启动识别失败: {e}")
                    asyncio.run_coroutine_threadsafe(
                        self.result_queue.put({"type": "error", "message": str(e)}),
                        self._loop
                    )
            
            thread = threading.Thread(target=start_recognition, daemon=True)
            thread.start()
            
            # 等待连接建立
            try:
                result = await asyncio.wait_for(self.result_queue.get(), timeout=10.0)
                if result.get("type") == "open":
                    self.state = STTState.STREAMING
                    logger.info("Aliyun STT 流式识别已启动")
                    return True
                elif result.get("type") == "error":
                    raise Exception(result.get("message", "Unknown error"))
            except asyncio.TimeoutError:
                raise Exception("连接超时")
            
        except Exception as e:
            self.state = STTState.ERROR
            logger.error(f"启动 STT 失败: {e}")
            raise
        
        return False
    
    async def send_audio(self, audio_chunk: bytes):
        """
        发送音频块
        
        Args:
            audio_chunk: PCM 音频数据（建议每包 100ms，约 3.2KB @ 16kHz 16bit mono）
        """
        if self.state != STTState.STREAMING:
            logger.warning(f"无法发送音频，当前状态: {self.state}")
            return
        
        if not self.recognition:
            return
        
        try:
            # send_audio_frame 是同步的，在线程池中执行以避免阻塞
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.recognition.send_audio_frame,
                audio_chunk
            )
        except Exception as e:
            logger.error(f"发送音频失败: {e}")
    
    async def get_result(self, timeout: float = 0.01) -> Optional[dict]:
        """
        获取识别结果（非阻塞）
        
        Returns:
            识别结果字典，或 None
        """
        if not self.result_queue:
            return None
        
        try:
            result = await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return None
    
    async def stop(self) -> str:
        """
        停止识别并获取最终结果
        
        Returns:
            完整的识别文本
        """
        if self.state not in (STTState.STREAMING, STTState.STARTING):
            return self._final_text
        
        self.state = STTState.STOPPING
        
        try:
            if self.recognition:
                # stop() 是阻塞的，在线程池中执行
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.recognition.stop)
            
            # 等待完成信号
            while True:
                try:
                    result = await asyncio.wait_for(self.result_queue.get(), timeout=5.0)
                    if result.get("type") == "complete":
                        self._final_text = result.get("text", "")
                        break
                    elif result.get("type") == "error":
                        logger.error(f"STT 错误: {result.get('message')}")
                        break
                    elif result.get("type") == "close":
                        break
                    elif result.get("type") == "sentence":
                        # 累积句子
                        pass
                except asyncio.TimeoutError:
                    logger.warning("等待 STT 完成超时")
                    break
            
            self.state = STTState.COMPLETED
            
        except Exception as e:
            self.state = STTState.ERROR
            logger.error(f"停止 STT 失败: {e}")
        
        return self._final_text
    
    async def cancel(self):
        """取消识别"""
        self.state = STTState.CANCELLED
        
        if self.recognition:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.recognition.stop)
            except Exception as e:
                logger.warning(f"取消 STT 时发生错误: {e}")
    
    def get_state(self) -> STTState:
        """获取当前状态"""
        return self.state


# 便捷函数：创建配置
def create_stt_config(
    api_key: str,
    model: str = "fun-asr-realtime",
    sample_rate: int = 16000,
    format: str = "pcm",
    vocabulary_id: Optional[str] = None
) -> STTConfig:
    """创建 STT 配置"""
    return STTConfig(
        api_key=api_key,
        model=model,
        sample_rate=sample_rate,
        format=format,
        vocabulary_id=vocabulary_id
    )