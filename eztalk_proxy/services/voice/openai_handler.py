import logging
import httpx
import json
from fastapi import HTTPException
from eztalk_proxy.core.http_client import get_http_client

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
    
    # 使用全局客户端
    client = get_http_client()
    
    # 智能补全 URL: 如果未包含 /transcriptions，则自动追加 /audio/transcriptions
    target_url = api_url
    if not target_url.endswith("/transcriptions"):
        target_url = f"{target_url.rstrip('/')}/audio/transcriptions"
        
    logger.info(f"Speech-to-Text using OpenAI Whisper at {target_url}")
    
    # OpenAI 需要文件名
    files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
    data = {'model': model}
    
    try:
        stt_resp = await client.post(target_url, headers=headers, files=files, data=data, timeout=30.0)
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
        
    # 智能补全 URL: 如果未包含 /completions，则自动追加 /chat/completions
    target_url = api_url
    if not target_url.endswith("/completions"):
        target_url = f"{target_url.rstrip('/')}/chat/completions"

    logger.info(f"Generate response using OpenAI {model} at {target_url}")
    
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
    
    # 使用全局客户端
    client = get_http_client()
    try:
        chat_resp = await client.post(
            target_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=chat_payload,
            timeout=30.0
        )
        if chat_resp.status_code != 200:
            logger.error(f"OpenAI Chat failed: {chat_resp.text}")
            raise Exception(f"OpenAI Chat failed: {chat_resp.status_code}")
            
        assistant_text = chat_resp.json()['choices'][0]['message']['content'].strip()
        logger.info(f"Chat Response: {assistant_text[:100]}...")
        return assistant_text
    except Exception as e:
        logger.exception("OpenAI Chat error")
async def process_tts(
    text: str,
    api_key: str,
    voice_name: str = "alloy",
    model: str = "tts-1",
    api_url: str = None
) -> bytes:
    """
    OpenAI Text-to-Speech
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API Key 未提供")
        
    if not api_url:
        raise HTTPException(status_code=400, detail="OpenAI API 地址未提供")
        
    logger.info(f"Text-to-Speech using OpenAI {model} ({voice_name}) at {api_url}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "input": text,
        "voice": voice_name,
        "response_format": "wav"  # Explicitly request WAV
    }
    
    client = get_http_client()
    try:
        # OpenAI TTS endpoint is usually /v1/audio/speech
        # We assume api_url is the full endpoint or base url. 
        # If the user provides base url (e.g. https://api.openai.com/v1), we might need to append /audio/speech if not present.
        # However, for consistency with other handlers, we expect the full URL or handle it here.
        # Given existing patterns, let's assume api_url is the full endpoint if it ends with /speech, otherwise append it.
        
        target_url = api_url
        if not target_url.endswith("/speech"):
             target_url = f"{target_url.rstrip('/')}/audio/speech"
             
        resp = await client.post(
            target_url,
            headers=headers,
            json=payload,
            timeout=60.0
        )
        
        if resp.status_code != 200:
            logger.error(f"OpenAI TTS failed: {resp.text}")
            raise Exception(f"OpenAI TTS failed: {resp.status_code} - {resp.text}")
            
        return resp.content
    except Exception as e:
        logger.exception("OpenAI TTS error")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")