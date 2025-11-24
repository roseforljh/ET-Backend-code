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