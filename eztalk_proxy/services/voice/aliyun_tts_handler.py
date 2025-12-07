"""
阿里云 Qwen-TTS 语音合成处理器

基于 DashScope Python SDK 实现 HTTP API 语音合成（非实时模式）。

官方文档：https://help.aliyun.com/zh/model-studio/developer-reference/qwen-tts

支持的模型：
- qwen3-tts-flash（推荐，支持 49 种音色，多语言）
- qwen-tts 系列

音频格式：
- PCM 24000Hz 16bit Mono (流式输出 Base64 编码)
- WAV 24000Hz (非流式输出)

相比 qwen3-tts-flash-realtime (WebSocket 模式)，HTTP API 模式更稳定：
- 无需维护 WebSocket 连接
- 更简单的错误处理和重试机制
- 适合并发请求场景
"""

import logging
import base64
import asyncio
from typing import AsyncGenerator, Optional, Generator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("EzTalkProxy.Services.Voice.AliyunTTS")

# 线程池用于在异步上下文中执行同步 SDK 调用
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="aliyun_tts_")

# 尝试导入 DashScope SDK
try:
    import dashscope
    from dashscope import MultiModalConversation
    _DASHSCOPE_TTS_AVAILABLE = True
except ImportError:
    _DASHSCOPE_TTS_AVAILABLE = False
    logger.warning("DashScope TTS SDK 不可用，请安装: pip install dashscope>=1.24.6")


@dataclass
class TTSConfig:
    """TTS 配置"""
    api_key: str
    api_url: str = "https://dashscope.aliyuncs.com/api/v1"
    model: str = "qwen3-tts-flash"
    voice: str = "Cherry"
    language_type: str = "Auto"  # Auto, Chinese, English, German, etc.
    sample_rate: int = 24000


# 支持的语言类型映射
LANGUAGE_TYPE_MAP = {
    "auto": "Auto",
    "chinese": "Chinese",
    "english": "English",
    "german": "German",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "japanese": "Japanese",
    "korean": "Korean",
    "french": "French",
    "russian": "Russian",
}


def _validate_model_name(model: str) -> str:
    """
    验证模型名称是否为非实时模型
    
    Args:
        model: 传入的模型名称
        
    Returns:
        验证后的模型名称
        
    Raises:
        ValueError: 如果使用了实时模型名称
    """
    if not model:
        return "qwen3-tts-flash"
    
    # 检查是否使用了实时模型（不再自动转换，直接报错）
    if "realtime" in model.lower():
        raise ValueError(
            f"不支持实时模型 '{model}'。请使用非实时模型，如 'qwen3-tts-flash'。"
            f"\n推荐模型: qwen3-tts-flash, qwen3-tts-flash-2025-11-27, qwen-tts"
        )
    
    return model


def _validate_api_url(api_url: str) -> str:
    """
    验证 API URL 是否为 HTTP URL
    
    Args:
        api_url: 传入的 API URL
        
    Returns:
        验证后的 API URL
        
    Raises:
        ValueError: 如果使用了 WebSocket URL
    """
    if not api_url:
        return "https://dashscope.aliyuncs.com/api/v1"
    
    # 检查是否使用了 WebSocket URL（不再自动转换，直接报错）
    if api_url.startswith("wss://") or api_url.startswith("ws://"):
        raise ValueError(
            f"不支持 WebSocket URL '{api_url}'。请使用 HTTP API URL。"
            f"\n正确的 URL: https://dashscope.aliyuncs.com/api/v1"
        )
    
    # 检查是否包含实时 API 路径
    if "/api-ws/" in api_url or "/realtime" in api_url:
        raise ValueError(
            f"不支持实时 API URL '{api_url}'。请使用 HTTP API URL。"
            f"\n正确的 URL: https://dashscope.aliyuncs.com/api/v1"
        )
    
    return api_url


def _detect_language_type(text: str) -> str:
    """
    简单的语言检测，用于设置 language_type 参数
    
    Args:
        text: 待合成的文本
        
    Returns:
        语言类型字符串
    """
    # 统计中文字符数量
    chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    # 统计英文字符数量
    english_count = sum(1 for char in text if char.isalpha() and ord(char) < 128)
    
    total = chinese_count + english_count
    if total == 0:
        return "Auto"
    
    # 如果中文占比超过 50%，返回 Chinese
    if chinese_count / total > 0.5:
        return "Chinese"
    elif english_count / total > 0.5:
        return "English"
    else:
        return "Auto"


def _process_tts_stream_sync(
    text: str,
    api_key: str,
    api_url: str = None,
    model: str = "qwen3-tts-flash",
    voice: str = "Cherry",
    language_type: str = "Auto",
) -> Generator[bytes, None, None]:
    """
    同步流式 TTS 合成
    
    按照官方文档示例实现：
    https://help.aliyun.com/zh/model-studio/developer-reference/qwen-tts
    
    Args:
        text: 待合成文本
        api_key: API Key
        api_url: API 地址
        model: 模型名称
        voice: 音色名称
        language_type: 语言类型
        
    Yields:
        PCM 音频数据块
    """
    if not _DASHSCOPE_TTS_AVAILABLE:
        raise RuntimeError("DashScope TTS SDK 不可用，请安装: pip install dashscope>=1.24.6")
    
    # 验证模型名称（不支持实时模型）
    model = _validate_model_name(model)
    
    # 验证 API URL（不支持 WebSocket URL）
    api_url = _validate_api_url(api_url)
    
    # 设置 API URL（必须在调用前设置）
    dashscope.base_http_api_url = api_url
    
    logger.info(f"Aliyun TTS: 开始合成 model={model}, voice={voice}, lang={language_type}, text_len={len(text)}, text='{text[:50]}...'")
    
    try:
        # 使用 dashscope.MultiModalConversation.call() 进行流式 TTS
        # 严格按照官方文档的参数格式
        response = dashscope.MultiModalConversation.call(
            model=model,
            api_key=api_key,
            text=text,
            voice=voice,
            language_type=language_type,
            stream=True
        )
        
        chunk_count = 0
        total_bytes = 0
        api_error = None
        
        for chunk in response:
            # 检查是否有错误
            if hasattr(chunk, 'status_code') and chunk.status_code != 200:
                error_msg = getattr(chunk, 'message', 'Unknown error')
                status_code = chunk.status_code
                logger.error(f"Aliyun TTS: API 错误 - status_code={status_code}, message={error_msg}")
                # 保存错误信息，稍后抛出异常
                api_error = RuntimeError(f"API error {status_code}: {error_msg}")
                break
            
            if chunk.output is not None:
                audio = chunk.output.audio
                
                # 尝试多种方式获取音频数据
                audio_data = None
                if audio is not None:
                    # 方式1: audio.data 属性
                    if hasattr(audio, 'data') and audio.data:
                        audio_data = audio.data
                    # 方式2: 字典访问
                    elif isinstance(audio, dict) and 'data' in audio:
                        audio_data = audio['data']
                
                if audio_data:
                    # 解码 Base64 音频数据
                    try:
                        pcm_data = base64.b64decode(audio_data)
                        chunk_count += 1
                        total_bytes += len(pcm_data)
                        yield pcm_data
                    except Exception as decode_error:
                        logger.warning(f"Aliyun TTS: Base64 解码失败: {decode_error}")
                
                # 检查是否完成
                finish_reason = getattr(chunk.output, 'finish_reason', None)
                if finish_reason == "stop":
                    logger.info(f"Aliyun TTS: 合成完成 - {chunk_count} 块, {total_bytes} 字节")
                    break
        
        # 如果有 API 错误，抛出异常以触发重试
        if api_error is not None:
            raise api_error
        
        # 如果没有收到任何音频数据，记录警告并抛出异常
        if chunk_count == 0:
            error_msg = f"Aliyun TTS: 未收到任何音频数据！model={model}, voice={voice}, text_len={len(text)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
                    
    except Exception as e:
        logger.error(f"Aliyun TTS 流式合成错误: {e}", exc_info=True)
        raise


def _process_tts_sync(
    text: str,
    api_key: str,
    api_url: str = None,
    model: str = "qwen3-tts-flash",
    voice: str = "Cherry",
    language_type: str = "Auto",
) -> bytes:
    """
    同步非流式 TTS 合成
    
    Args:
        text: 待合成文本
        api_key: API Key
        api_url: API 地址
        model: 模型名称
        voice: 音色名称
        language_type: 语言类型
        
    Returns:
        完整的 PCM 音频数据
    """
    if not _DASHSCOPE_TTS_AVAILABLE:
        raise RuntimeError("DashScope TTS SDK 不可用，请安装: pip install dashscope>=1.24.6")
    
    # 验证模型名称（不支持实时模型）
    model = _validate_model_name(model)
    
    # 验证 API URL（不支持 WebSocket URL）
    api_url = _validate_api_url(api_url)
    
    # 设置 API URL
    dashscope.base_http_api_url = api_url
    
    logger.info(f"Aliyun TTS: 开始合成 (非流式) model={model}, voice={voice}, url={api_url}, text='{text[:30]}...'")
    
    try:
        # 使用 MultiModalConversation.call() 进行非流式 TTS
        response = MultiModalConversation.call(
            model=model,
            api_key=api_key,
            text=text,
            voice=voice,
            language_type=language_type,
            stream=False
        )
        
        # 非流式模式返回音频 URL
        if response.status_code == 200:
            audio = response.output.audio
            if audio is not None:
                # 如果有 data 字段，直接解码
                if audio.data:
                    pcm_data = base64.b64decode(audio.data)
                    logger.info(f"Aliyun TTS: 合成完成 (非流式) - {len(pcm_data)} 字节")
                    return pcm_data
                # 如果有 url 字段，需要下载（这里暂时不实现，流式模式更适合）
                elif hasattr(audio, 'url') and audio.url:
                    logger.warning(f"Aliyun TTS: 返回了 URL 而非数据，URL={audio.url}")
                    # 可以通过 httpx 下载，但这里推荐使用流式模式
                    raise ValueError("非流式模式返回 URL，建议使用流式模式")
        
        error_msg = f"TTS 请求失败: status_code={response.status_code}"
        if hasattr(response, 'message'):
            error_msg += f", message={response.message}"
        raise Exception(error_msg)
        
    except Exception as e:
        logger.error(f"Aliyun TTS 非流式合成错误: {e}")
        raise


async def process_tts_stream(
    text: str,
    api_key: str,
    api_url: str = None,
    model: str = "qwen3-tts-flash",
    voice: str = "Cherry",
    sample_rate: int = 24000,
    language_type: str = None,
    **kwargs
) -> AsyncGenerator[bytes, None]:
    """
    阿里云 TTS 流式语音合成 (异步接口)
    
    Args:
        text: 待合成文本
        api_key: 阿里云 API Key
        api_url: API 地址（可选）
        model: 模型名称
        voice: 音色名称
        sample_rate: 采样率（qwen3-tts-flash 固定 24kHz）
        language_type: 语言类型（可选，自动检测）
        
    Yields:
        PCM 音频数据块
    """
    if not _DASHSCOPE_TTS_AVAILABLE:
        raise RuntimeError("DashScope TTS SDK 不可用，请安装: pip install dashscope>=1.24.6")
    
    # 自动检测语言类型
    if language_type is None:
        language_type = _detect_language_type(text)
    
    # 在线程池中执行同步生成器
    loop = asyncio.get_event_loop()
    
    # 使用队列在线程和协程之间传递数据
    queue: asyncio.Queue = asyncio.Queue()
    
    def run_sync_generator():
        """在线程中运行同步生成器，将结果放入队列"""
        try:
            for chunk in _process_tts_stream_sync(
                text=text,
                api_key=api_key,
                api_url=api_url,
                model=model,
                voice=voice,
                language_type=language_type
            ):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            # 发送完成信号
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(e), loop)
    
    # 提交到线程池
    future = loop.run_in_executor(_executor, run_sync_generator)
    
    # 从队列中读取数据
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=60.0)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        except asyncio.TimeoutError:
            logger.warning("Aliyun TTS: 读取音频数据超时")
            break
    
    # 等待线程完成
    await future


async def process_tts(
    text: str,
    api_key: str,
    api_url: str = None,
    model: str = "qwen3-tts-flash",
    voice: str = "Cherry",
    sample_rate: int = 24000,
    language_type: str = None,
    **kwargs
) -> bytes:
    """
    阿里云 TTS 语音合成（非流式，返回完整音频）
    
    Args:
        text: 待合成文本
        api_key: 阿里云 API Key
        api_url: API 地址（可选）
        model: 模型名称
        voice: 音色名称
        sample_rate: 采样率
        language_type: 语言类型（可选，自动检测）
        
    Returns:
        PCM 音频数据
    """
    if not _DASHSCOPE_TTS_AVAILABLE:
        raise RuntimeError("DashScope TTS SDK 不可用，请安装: pip install dashscope>=1.24.6")
    
    # 自动检测语言类型
    if language_type is None:
        language_type = _detect_language_type(text)
    
    # 使用流式模式收集所有数据（更可靠）
    audio_chunks = []
    async for chunk in process_tts_stream(
        text=text,
        api_key=api_key,
        api_url=api_url,
        model=model,
        voice=voice,
        sample_rate=sample_rate,
        language_type=language_type
    ):
        audio_chunks.append(chunk)
    
    return b"".join(audio_chunks)


# 便捷函数：创建 TTS 配置
def create_tts_config(
    api_key: str,
    model: str = "qwen3-tts-flash",
    voice: str = "Cherry",
    api_url: str = None,
    sample_rate: int = 24000,
    language_type: str = "Auto"
) -> TTSConfig:
    """创建 TTS 配置"""
    return TTSConfig(
        api_key=api_key,
        api_url=api_url or "https://dashscope.aliyuncs.com/api/v1",
        model=model,
        voice=voice,
        language_type=language_type,
        sample_rate=sample_rate
    )


# 支持的音色列表 (qwen3-tts-flash)
SUPPORTED_VOICES = {
    # 中文女声
    "Cherry": {"gender": "female", "language": "Chinese", "description": "温柔甜美"},
    "Serena": {"gender": "female", "language": "Chinese", "description": "知性优雅"},
    "Bailing": {"gender": "female", "language": "Chinese", "description": "活泼开朗"},
    "Chelsie": {"gender": "female", "language": "Chinese", "description": "清新自然"},
    "Ethan": {"gender": "male", "language": "Chinese", "description": "沉稳大气"},
    "Aidan": {"gender": "male", "language": "Chinese", "description": "青年活力"},
    # 英文
    "Emily": {"gender": "female", "language": "English", "description": "American English"},
    "Lily": {"gender": "female", "language": "English", "description": "British English"},
    # ... 更多音色请参考官方文档
}