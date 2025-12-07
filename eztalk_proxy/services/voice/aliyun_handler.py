import logging
import tempfile
import time
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException

logger = logging.getLogger("EzTalkProxy.Services.Voice.Aliyun")

# 尝试导入 DashScope SDK
try:
    from dashscope.audio.asr import Recognition, RecognitionCallback
    _DASHSCOPE_AVAILABLE = True
except ImportError:
    _DASHSCOPE_AVAILABLE = False
    logger.warning("DashScope SDK 不可用，请安装: pip install dashscope>=1.23.1")

# 全局线程池，避免每次请求都创建新线程
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aliyun_stt_")

def check_dashscope_available():
    """检查 DashScope SDK 是否可用"""
    if not _DASHSCOPE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="DashScope SDK 不可用，请在后端安装: pip install dashscope>=1.23.1"
        )

def _sync_stt_call(
    audio_bytes: bytes,
    api_key: str,
    model: str,
    format: str,
    sample_rate: int,
    vocabulary_id: str = None
) -> str:
    """
    同步执行 STT 调用（在线程池中运行）
    
    优化点：
    - 使用内存缓冲区而非临时文件（如果 SDK 支持）
    - 添加详细延迟日志
    """
    start_time = time.time()
    
    # 创建识别对象
    recognition_params = {
        "model": model,
        "format": format,
        "sample_rate": sample_rate,
        "api_key": api_key
    }
    
    if vocabulary_id:
        recognition_params["vocabulary_id"] = vocabulary_id
    
    class DummyCallback(RecognitionCallback):
        def on_open(self): pass
        def on_close(self): pass
        def on_complete(self): pass
        def on_error(self, result): pass
        def on_event(self, result): pass

    recognition = Recognition(
        callback=DummyCallback(),
        **recognition_params
    )
    
    init_time = time.time()
    logger.info(f"[Latency] SDK 初始化耗时: {(init_time - start_time)*1000:.0f}ms")
    
    # 写入临时文件
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_path = temp_file.name
        
        file_time = time.time()
        logger.info(f"[Latency] 临时文件写入耗时: {(file_time - init_time)*1000:.0f}ms")
        
        # 调用 SDK
        result = recognition.call(file=temp_path)
        
        call_time = time.time()
        logger.info(f"[Latency] SDK call() 耗时: {(call_time - file_time)*1000:.0f}ms")
        
        if result.status_code == 200:
            if hasattr(result, 'get_sentence'):
                sentence = result.get_sentence()
                
                if isinstance(sentence, dict):
                    text = sentence.get('text', '').strip()
                elif isinstance(sentence, list) and len(sentence) > 0:
                    text = ' '.join([s.get('text', '') for s in sentence]).strip()
                else:
                    text = ''
                
                total_time = time.time()
                logger.info(f"[Latency] STT 总耗时: {(total_time - start_time)*1000:.0f}ms, 文本长度: {len(text)}")
                return text
            else:
                logger.warning("STT success but no sentence found")
                return ""
        else:
            logger.error(f"Aliyun STT failed: {result.code} - {result.message}")
            raise Exception(f"Aliyun STT failed: {result.message}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

async def process_stt(
    audio_bytes: bytes,
    api_key: str,
    api_url: str = None,
    model: str = "fun-asr-realtime",
    format: str = "opus",
    sample_rate: int = 16000,
    vocabulary_id: str = None,
    **kwargs
) -> str:
    """
    阿里云Fun-ASR语音识别（非实时）
    
    优化：
    - 使用线程池执行阻塞调用，避免阻塞事件循环
    - 添加详细延迟日志，便于诊断性能瓶颈
    
    Args:
        audio_bytes: 音频数据（Opus/WAV等格式）
        api_key: 用户配置的阿里云API Key
        api_url: 可选，自定义API地址（保留接口一致性）
        model: 用户指定的模型名称（如 fun-asr-realtime）
        format: 音频格式，默认opus
        sample_rate: 采样率，默认16000Hz
        vocabulary_id: 可选热词表ID
    
    Returns:
        识别的文本结果
    """
    check_dashscope_available()
    
    if not api_key:
        raise HTTPException(status_code=400, detail="阿里云API Key未配置")
    
    if not model:
        raise HTTPException(status_code=400, detail="STT模型名称未配置")
    
    logger.info(f"Aliyun STT 开始识别: model={model}, format={format}, rate={sample_rate}Hz, audio_size={len(audio_bytes)}bytes")
    
    try:
        # 在线程池中执行阻塞调用
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            _executor,
            _sync_stt_call,
            audio_bytes,
            api_key,
            model,
            format,
            sample_rate,
            vocabulary_id
        )
        return text
        
    except Exception as e:
        logger.exception("Aliyun STT error")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")