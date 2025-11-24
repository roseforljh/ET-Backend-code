import os
import logging
import httpx
import wave
import io
from typing import Optional

logger = logging.getLogger("EzTalkProxy.Services.Voice.Minimax")

def wave_file_bytes(pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> bytes:
    """将PCM数据转换为WAV格式的字节流"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    buffer.seek(0)
    return buffer.read()

async def synthesize_minimax_t2a(text: str, voice_id: str = "male-qn-qingse", api_url: str = None, api_key: str = None) -> tuple[Optional[bytes], int]:
    """
    调用 Minimax T2A 接口进行语音合成
    返回: (wav_bytes, sample_rate)
    """
    # 优先使用传入的 api_key，否则使用环境变量
    final_api_key = api_key or os.getenv("MINIMAX_T2A_API_KEY")
    
    if not final_api_key:
        logger.warning("Minimax API Key not provided (neither in args nor env), skipping Minimax TTS")
        return None, 24000
        
    if not api_url:
        logger.error("Minimax API URL not provided")
        return None, 24000
        
    url = api_url.strip() # 去除首尾空格
    
    logger.info(f"Minimax T2A Request URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json"
    }
    
    # 默认参数 (参考 Minimax 官方文档)
    payload = {
        "model": "speech-2.6-hd",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
            "emotion": "happy" # 自动匹配情绪，也可指定
        },
        "audio_setting": {
            "sample_rate": 24000, # 支持 32000, 24000 等
            "bitrate": 128000,
            "format": "pcm", # 使用 pcm 方便后续转 wav
            "channel": 1
        },
        "subtitle_enable": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Minimax T2A failed: {response.status_code} - {response.text}")
                return None, 24000
                
            resp_json = response.json()
            base_resp = resp_json.get("base_resp", {})
            if base_resp.get("status_code") != 0:
                logger.error(f"Minimax T2A error: {base_resp.get('status_msg')}")
                return None, 24000
                
            data = resp_json.get("data", {})
            audio_hex = data.get("audio")
            if not audio_hex:
                logger.error("Minimax T2A returned no audio data")
                return None, 24000
                
            # Hex 转 PCM bytes
            pcm_bytes = bytes.fromhex(audio_hex)
            
            # 获取实际采样率
            extra_info = resp_json.get("extra_info", {})
            sample_rate = extra_info.get("audio_sample_rate", 24000)
            
            # 包装成 WAV
            wav_bytes = wave_file_bytes(pcm_bytes, rate=sample_rate)
            
            return wav_bytes, sample_rate
            
    except Exception as e:
        logger.exception(f"Minimax T2A exception: {e}")
        return None, 24000