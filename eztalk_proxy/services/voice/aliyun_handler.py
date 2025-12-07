import logging
import tempfile
from fastapi import HTTPException

logger = logging.getLogger("EzTalkProxy.Services.Voice.Aliyun")

# 尝试导入 DashScope SDK
try:
    from dashscope.audio.asr import Recognition, RecognitionCallback
    _DASHSCOPE_AVAILABLE = True
except ImportError:
    _DASHSCOPE_AVAILABLE = False
    logger.warning("DashScope SDK 不可用，请安装: pip install dashscope>=1.23.1")

def check_dashscope_available():
    """检查 DashScope SDK 是否可用"""
    if not _DASHSCOPE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="DashScope SDK 不可用，请在后端安装: pip install dashscope>=1.23.1"
        )

async def process_stt(
    audio_bytes: bytes,
    api_key: str,
    api_url: str = None,      # 可选，用户自定义API地址（DashScope SDK可能不支持，保留接口一致性）
    model: str = "fun-asr-realtime",
    format: str = "opus",     # 默认Opus格式
    sample_rate: int = 16000,
    vocabulary_id: str = None,  # 可选：热词ID
    **kwargs
) -> str:
    """
    阿里云Fun-ASR语音识别（非流式）
    
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
    
    # 调试日志：显示接收到的参数
    logger.info(f"Aliyun STT 接收参数: api_key={'已提供' if api_key else '未提供'}, model={model}, format={format}, api_url={api_url}")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="阿里云API Key未配置")
    
    if not model:
        raise HTTPException(status_code=400, detail="STT模型名称未配置")
    
    logger.info(f"Aliyun STT 开始识别: model={model}, format={format}, rate={sample_rate}Hz, audio_size={len(audio_bytes)}bytes")
    
    try:
        # 创建识别对象
        recognition_params = {
            "model": model,
            "format": format,
            "sample_rate": sample_rate,
            "api_key": api_key
        }
        
        # 添加可选参数
        if vocabulary_id:
            recognition_params["vocabulary_id"] = vocabulary_id
        
        # 即使是同步调用，部分SDK版本仍要求实例化时传入callback
        # 创建一个 Dummy Callback
        class DummyCallback(RecognitionCallback):
            def on_open(self): pass
            def on_close(self): pass
            def on_complete(self): pass
            def on_error(self, result): pass
            def on_event(self, result): pass

        # 必须传入 callback
        recognition = Recognition(
            callback=DummyCallback(),
            **recognition_params
        )
        
        # 同步调用（将bytes写入临时文件）
        # Windows下不能在文件打开时再次打开，所以需要先关闭文件
        import os
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                temp_path = temp_file.name
            
            # 使用实例方法 call()
            result = recognition.call(file=temp_path)
            
            if result.status_code == 200:
                if hasattr(result, 'get_sentence'):
                    sentence = result.get_sentence()
                    
                    # 处理单句结果
                    if isinstance(sentence, dict):
                        text = sentence.get('text', '').strip()
                    # 处理多句结果
                    elif isinstance(sentence, list) and len(sentence) > 0:
                        text = ' '.join([s.get('text', '') for s in sentence]).strip()
                    else:
                        text = ''
                    
                    logger.info(f"STT Result: {text[:100]}...")
                    return text
                else:
                    logger.warning("STT success but no sentence found")
                    return ""
            else:
                logger.error(f"Aliyun STT failed: {result.code} - {result.message}")
                raise Exception(f"Aliyun STT failed: {result.message}")
        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
                
    except Exception as e:
        logger.exception("Aliyun STT error")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")