import logging

logger = logging.getLogger("EzTalkProxy.Services.Voice.Validator")

def validate_voice_config(platform: str, voice_name: str) -> None:
    """
    校验 TTS 音色配置。
    如果配置无效，抛出 ValueError。
    
    Args:
        platform: TTS 平台名称 (Minimax, SiliconFlow, OpenAI, Gemini)
        voice_name: 音色名称/ID
        
    Raises:
        ValueError: 当音色无效时
    """
    if not voice_name:
        raise ValueError(f"Voice name is required for platform {platform}")
        
    normalized_platform = platform.lower()
    normalized_voice = voice_name.lower()
    
    # Gemini 默认音色是 "Kore"，如果其他平台收到了这个值，说明前端传错了
    is_default_gemini_voice = normalized_voice in ["kore", "default"]
    
    if normalized_platform != "gemini" and is_default_gemini_voice:
        msg = f"Invalid voice '{voice_name}' for platform {platform}. Please select a valid voice."
        logger.error(msg)
        raise ValueError(msg)
        
    # OpenAI 特定校验
    if normalized_platform == "openai":
        if is_default_gemini_voice: # Alloy 是默认的，但 Kore 肯定不对
             raise ValueError(f"Invalid OpenAI voice: '{voice_name}'")

    # Minimax 特定校验
    if normalized_platform == "minimax":
        if is_default_gemini_voice:
            raise ValueError(f"Invalid Minimax voice: '{voice_name}'")

    # SiliconFlow 特定校验
    if normalized_platform == "siliconflow":
        if is_default_gemini_voice:
            raise ValueError(f"Invalid SiliconFlow voice: '{voice_name}'")