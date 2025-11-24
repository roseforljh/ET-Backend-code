import logging
import httpx
from fastapi import HTTPException

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
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info(f"Speech-to-Text using SiliconFlow model '{model}' at {api_url}")
        
        # 构造 multipart/form-data
        files = {'file': (filename, audio_bytes, mime_type)}
        data = {'model': model}
        
        try:
            resp = await client.post(api_url, headers=headers, files=files, data=data)
            
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
    
    payload = {
        "model": model,
        "input": text,
        "voice": final_voice,
        "response_format": response_format,
        "sample_rate": sample_rate,
        "stream": False,
        "gain": 0
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(api_url, headers=headers, json=payload)
            
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
                
            return resp.content, sample_rate
            
        except Exception as e:
            logger.exception("SiliconFlow TTS error")
            raise HTTPException(status_code=500, detail=f"硅基流动语音合成失败: {str(e)}")

async def process_tts_stream(text: str, api_key: str, api_url: str, model: str, voice: str, response_format: str = "pcm", sample_rate: int = 32000):
    """
    调用硅基流动 (SiliconFlow) 的 TTS 接口进行流式语音合成
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
    
    payload = {
        "model": model,
        "input": text,
        "voice": final_voice,
        "response_format": response_format,
        "sample_rate": sample_rate,
        "stream": True, # 开启流式
        "gain": 0
    }
    
    logger.info(f"Starting SiliconFlow Stream TTS: {model} ({final_voice})")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", api_url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    logger.error(f"SiliconFlow Stream TTS failed: {response.status_code}")
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