import logging
import httpx
import json
from fastapi import HTTPException

logger = logging.getLogger("EzTalkProxy.Services.Voice.OpenAI")

async def process_stt(audio_bytes: bytes, api_key: str, api_url: str = None, model: str = "whisper-1") -> str:
    """
    OpenAI STT (Whisper)
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API Key 未提供")
        
    if not api_url:
        raise HTTPException(status_code=400, detail="OpenAI API 地址未提供")
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info(f"Speech-to-Text using OpenAI Whisper at {api_url}")
        
        # OpenAI 需要文件名
        files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
        data = {'model': model}
        
        try:
            stt_resp = await client.post(api_url, headers=headers, files=files, data=data)
            if stt_resp.status_code != 200:
                logger.error(f"OpenAI STT failed: {stt_resp.text}")
                raise Exception(f"OpenAI STT failed: {stt_resp.status_code}")
            
            text = stt_resp.json().get("text", "").strip()
            logger.info(f"STT Result: {text[:100]}...")
            return text
        except Exception as e:
            logger.exception("OpenAI STT error")
            raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")

async def process_chat(
    user_text: str, 
    chat_history: list, 
    system_prompt: str, 
    api_key: str, 
    api_url: str = None, 
    model: str = "gpt-4o-mini"
) -> str:
    """
    OpenAI Chat Completion
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API Key 未提供")
        
    if not api_url:
        raise HTTPException(status_code=400, detail="OpenAI API 地址未提供")
        
    logger.info(f"Generate response using OpenAI {model} at {api_url}")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # 只保留最近几轮
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent_history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": user_text})
    
    chat_payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 150
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            chat_resp = await client.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=chat_payload
            )
            if chat_resp.status_code != 200:
                logger.error(f"OpenAI Chat failed: {chat_resp.text}")
                raise Exception(f"OpenAI Chat failed: {chat_resp.status_code}")
                
            assistant_text = chat_resp.json()['choices'][0]['message']['content'].strip()
            logger.info(f"Chat Response: {assistant_text[:100]}...")
            return assistant_text
        except Exception as e:
            logger.exception("OpenAI Chat error")
            raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")