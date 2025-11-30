import logging
from fastapi import HTTPException

logger = logging.getLogger("EzTalkProxy.Services.Voice.Google")

# 尝试导入 google-genai
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

def check_genai_available():
    if not _GENAI_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="google-genai SDK 不可用，请在后端安装：pip install google-genai"
        )

def get_client(api_key: str, api_url: str = None):
    check_genai_available()
    if not api_key:
        # raise HTTPException(status_code=400, detail="Google API Key 未提供")
        pass # 允许空 Key，某些中转可能不需要 Key
        
    client_params = {"api_key": api_key}
    if api_url:
        # Pydantic 校验严格，HttpOptions 字段通常为 base_url (或尝试兼容 api_endpoint 如果版本允许，但报错说 forbidden)
        # 尝试使用 base_url 替代 api_endpoint
        client_params["http_options"] = {"base_url": api_url}
        
    return genai.Client(**client_params)

def process_stt(audio_bytes: bytes, mime_type: str, api_key: str, model: str, api_url: str = None) -> str:
    """
    Google Gemini STT
    """
    client = get_client(api_key, api_url)
    logger.info(f"Speech-to-Text using Gemini {model}")
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                ),
                "转录："
            ]
        )
        text = response.text.strip() if response.text else ""
        logger.info(f"STT Result: {text[:100]}...")
        return text
    except Exception as e:
        logger.exception("Gemini STT error")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")

def process_chat(user_text: str, chat_history: list, system_prompt: str, api_key: str, model: str, api_url: str = None) -> str:
    """
    Google Gemini Chat
    """
    client = get_client(api_key, api_url)
    logger.info(f"Generate response using Gemini {model}")
    
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    
    # 只保留最近几轮
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent_history:
        prompt_parts.append(msg.get("content", ""))
    
    prompt_parts.append(user_text)
    full_prompt = "\n".join(prompt_parts)
    
    try:
        chat_response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=150
            )
        )
        
        assistant_text = chat_response.text.strip() if chat_response.text else ""
        logger.info(f"Chat Response: {assistant_text[:100]}...")
        return assistant_text
    except Exception as e:
        logger.exception("Gemini Chat error")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")

def process_chat_stream(user_text: str, chat_history: list, system_prompt: str, api_key: str, model: str, api_url: str = None):
    """
    Google Gemini Chat (Streaming)
    Yields chunks of text.
    """
    client = get_client(api_key, api_url)
    logger.info(f"Stream response using Gemini {model}")
    
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt)
    
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in recent_history:
        prompt_parts.append(msg.get("content", ""))
    
    prompt_parts.append(user_text)
    full_prompt = "\n".join(prompt_parts)
    
    try:
        chat_response = client.models.generate_content_stream(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=150
            )
        )
        
        for chunk in chat_response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.exception("Gemini Chat Stream error")
        yield f"Error: {str(e)}"

def process_tts(text: str, api_key: str, voice_name: str, model: str, api_url: str = None) -> bytes:
    """
    Google Gemini TTS
    """
    client = get_client(api_key, api_url)
    logger.info(f"Text-to-Speech using Gemini {model} (Voice: {voice_name})")
    
    # 如果回复太长，截断避免TTS超时（保留前200字符）
    text_for_tts = text[:200] if len(text) > 200 else text
    if len(text) > 200:
        logger.warning(f"AI response too long ({len(text)} chars), truncated to 200 for TTS")
    
    try:
        tts_response = client.models.generate_content(
            model=model,
            contents=text_for_tts,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            )
        )
        
        # 提取音频数据（24kHz PCM）
        if tts_response.candidates and tts_response.candidates[0].content.parts:
             return tts_response.candidates[0].content.parts[0].inline_data.data
        return b""
    except Exception as e:
        logger.exception("Gemini TTS error")
        # TTS失败时返回空字节，允许降级为纯文本
        return b""