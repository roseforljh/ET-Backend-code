import logging
import httpx
from fastapi import HTTPException
from eztalk_proxy.core.http_client import get_http_client

logger = logging.getLogger("EzTalkProxy.Services.Voice.SiliconFlow")

async def process_stt(audio_bytes: bytes, api_key: str, api_url: str, model: str, mime_type: str = "audio/wav") -> str:
    """
    调用硅基流动 (SiliconFlow) 的 STT 接口进行语音转文本
    API文档参考: https://docs.siliconflow.cn/api-reference/audio/create-transcription
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="SiliconFlow API Key 未提供")
        
    if not api_url:
        # 默认官方地址，虽然前端会传，但后端做个兜底
        api_url = "https://api.siliconflow.cn/v1/audio/transcriptions"
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # 提取文件名后缀，虽然 API 可能不太关心，但保持规范
    ext = mime_type.split('/')[-1] if '/' in mime_type else 'wav'
    filename = f"audio.{ext}"
    
    # 使用全局客户端
    client = get_http_client()
    logger.info(f"Speech-to-Text using SiliconFlow model '{model}' at {api_url}")
    
    # 构造 multipart/form-data
    files = {'file': (filename, audio_bytes, mime_type)}
    data = {'model': model}
    
    try:
        resp = await client.post(api_url, headers=headers, files=files, data=data, timeout=30.0)
            
        if resp.status_code != 200:
            error_msg = resp.text
            logger.error(f"SiliconFlow STT failed: {resp.status_code} - {error_msg}")
            # 尝试提取更具体的错误信息
            try:
                err_json = resp.json()
                if "message" in err_json:
                    error_msg = err_json["message"]
            except:
                pass
            raise Exception(f"API Error ({resp.status_code}): {error_msg}")
        
        result = resp.json()
        text = result.get("text", "").strip()
        
        if not text:
            logger.warning("SiliconFlow STT returned empty text")
        else:
            logger.info(f"STT Result: {text[:100]}...")
            
        return text
        
    except Exception as e:
        logger.exception("SiliconFlow STT error")
        raise HTTPException(status_code=500, detail=f"硅基流动语音识别失败: {str(e)}")
async def process_tts(text: str, api_key: str, api_url: str, model: str, voice: str, response_format: str = "pcm", sample_rate: int = 32000) -> tuple[bytes, int]:
    """
    调用硅基流动 (SiliconFlow) 的 TTS 接口进行语音合成
    
    Args:
        response_format: 音频格式，支持 pcm, opus, mp3, wav。默认 pcm（Opus 存在解码兼容性问题）
        sample_rate: 采样率。默认 32000（PCM 格式，更高音质）
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="SiliconFlow API Key 未提供")
    
    if not api_url:
        api_url = "https://api.siliconflow.cn/v1/audio/speech"
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 处理 voice 参数，如果未包含模型前缀则自动添加
    # 例如 model="fishaudio/fish-speech-1.5", voice="alex" -> "fishaudio/fish-speech-1.5:alex"
    final_voice = voice
    if model and voice and ":" not in voice:
        final_voice = f"{model}:{voice}"
    
    # 使用配置的音频格式和采样率
    final_response_format = response_format
    final_sample_rate = sample_rate
    
    payload = {
        "model": model,
        "input": text,
        "voice": final_voice,
        "response_format": final_response_format,
        "sample_rate": final_sample_rate,
        "stream": False,
        "gain": 0
    }
    
    # 使用全局客户端
    client = get_http_client()
    try:
        resp = await client.post(api_url, headers=headers, json=payload, timeout=60.0)
            
        if resp.status_code != 200:
            error_msg = resp.text
            try:
                err_json = resp.json()
                if "message" in err_json:
                    error_msg = err_json["message"]
            except:
                pass
            logger.error(f"SiliconFlow TTS failed: {resp.status_code} - {error_msg}")
            raise Exception(f"API Error ({resp.status_code}): {error_msg}")
            
        return resp.content, final_sample_rate
        
    except Exception as e:
        logger.exception("SiliconFlow TTS error")
        raise HTTPException(status_code=500, detail=f"硅基流动语音合成失败: {str(e)}")

async def process_tts_stream(text: str, api_key: str, api_url: str, model: str, voice: str, response_format: str = "pcm", sample_rate: int = 32000):
    """
    调用硅基流动 (SiliconFlow) 的 TTS 接口进行流式语音合成
    
    Args:
        response_format: 音频格式，支持 pcm, opus, mp3, wav。默认 pcm（避免 Opus 解码兼容性问题）
        sample_rate: 采样率。默认 32000（PCM 格式，参考官方文档）
        
    Yields: bytes (chunk)
    """
    if not api_key:
        # raise HTTPException... yield 不能直接 raise HTTP Exception，记录日志并返回
        logger.error("SiliconFlow API Key missing for stream")
        return
    
    if not api_url:
        api_url = "https://api.siliconflow.cn/v1/audio/speech"
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    final_voice = voice
    if model and voice and ":" not in voice:
        final_voice = f"{model}:{voice}"
    
    # 使用配置的音频格式和采样率
    final_response_format = response_format
    final_sample_rate = sample_rate
    
    # IndexTTS-2 模型特殊处理：该模型在 32kHz 下可能产生爆音，降低到 24kHz
    if model and "IndexTTS-2" in model:
        final_sample_rate = 24000
        logger.info(f"IndexTTS-2 detected, using sample_rate=24000 to avoid audio artifacts")
    
    # IndexTTS-2 模型可能不支持流式输出，禁用 stream
    enable_stream = True
    if model and "IndexTTS-2" in model:
        enable_stream = False
        logger.info(f"IndexTTS-2 detected, disabling stream mode due to API limitations")
    
    payload = {
        "model": model,
        "input": text,
        "voice": final_voice,
        "response_format": final_response_format,
        "sample_rate": final_sample_rate,
        "stream": enable_stream
    }
    
    # gain 参数可能不被所有模型支持，仅在需要时添加
    # if "gain" in tts_config:
    #     payload["gain"] = tts_config["gain"]
    
    logger.info(f"Starting SiliconFlow Stream TTS: {model} ({final_voice}), format={final_response_format}, sample_rate={final_sample_rate}")
    
    try:
        # 使用全局客户端进行流式请求
        client = get_http_client()
        async with client.stream("POST", api_url, headers=headers, json=payload, timeout=60.0) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                logger.error(f"SiliconFlow Stream TTS failed: {response.status_code}")
                logger.error(f"Error details: {error_body.decode('utf-8', errors='ignore')}")
                logger.error(f"Request payload: {payload}")
                return

            chunk_count = 0
            total_bytes = 0
            
            async for chunk in response.aiter_bytes():
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk
                    
            logger.info(f"SiliconFlow Stream TTS completed. Chunks: {chunk_count}, Total bytes: {total_bytes}")
                        
    except Exception as e:
        logger.exception(f"SiliconFlow Stream TTS exception: {e}")