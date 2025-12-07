import logging

logger = logging.getLogger("EzTalkProxy.Services.Voice.Validator")

# 阿里云 TTS 支持的音色列表
ALIYUN_VOICES = {
    # 国内音色
    "cherry", "serena", "ethan", "chelsie", "momo", "vivian", "moon", "maia",
    "kai", "nofish", "bella", "eldric sage", "mia", "mochi", "bellona", "vincent",
    "bunny", "neil", "elias", "arthur", "nini", "ebona", "seren", "pip",
    "stella", "ryan", "andre", "jennifer",
    # 国外音色
    "aiden", "katerina", "bodega", "sonrisa", "alek", "dolce", "sohee",
    "ono anna", "lenn", "emilien", "radio gol",
    # 乡音音色
    "jada", "dylan", "li", "marcus", "roy", "peter", "sunny", "eric", "rocky", "kiki"
}

def validate_voice_config(platform: str, voice_name: str) -> None:
    """
    校验 TTS 音色配置。
    如果配置无效，抛出 ValueError。
    
    Args:
        platform: TTS 平台名称 (Minimax, SiliconFlow, OpenAI, Gemini, Aliyun)
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

    # Aliyun 特定校验
    if normalized_platform == "aliyun":
        if is_default_gemini_voice:
            raise ValueError(f"Invalid Aliyun voice: '{voice_name}'")
        # 校验是否是有效的阿里云音色
        if normalized_voice not in ALIYUN_VOICES:
            logger.warning(f"Aliyun voice '{voice_name}' not in predefined list, but allowing custom voice")