import asyncio
import base64
import io
import logging
import os
import tempfile
import wave
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger("EzTalkProxy.Routers.GeminiVoiceChat")
router = APIRouter()

# 使用新的 google-genai SDK（支持TTS）
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
    logger.info("Successfully imported google-genai SDK for TTS support")
except Exception as e:
    logger.error(f"Failed to import google-genai: {e}")
    _GENAI_AVAILABLE = False

DEFAULT_STT_MODEL = "gemini-flash-lite-latest"   # 最快的轻量级模型
DEFAULT_CHAT_MODEL = "gemini-flash-lite-latest"  # 对话也用lite，追求极致速度
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"  # 2.0不存在，用回2.5
DEFAULT_VOICE_NAME = "Kore"  # 默认音色（实际使用前端传递的值）


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
    import httpx
    
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


@router.post("/gemini/voice-chat/complete", summary="完整语音对话：STT → Chat → TTS", tags=["Gemini Voice Chat"])
async def complete_voice_chat(
    audio: UploadFile = File(..., description="用户录音文件（支持多种格式：wav, mp3, ogg, flac等）"),
    api_key: str = Form(..., description="Google AI Studio API Key"),
    chat_history: str = Form("[]", description="对话历史，JSON数组格式，例如：[{\"role\":\"user\",\"content\":\"你好\"},{\"role\":\"assistant\",\"content\":\"你好！\"}]"),
    system_prompt: str = Form("", description="系统提示词（可选）"),
    voice_name: str = Form(DEFAULT_VOICE_NAME, description="TTS语音名称（可选）"),
    voice_platform: str = Form("Gemini", description="TTS 平台：Gemini / Minimax 等"),
    provider_api_url: str = Form(None, description="大模型厂商 API 地址（可选，用于 Minimax 等）"),
    stt_model: str = Form(DEFAULT_STT_MODEL, description="语音识别模型"),
    chat_model: str = Form(DEFAULT_CHAT_MODEL, description="对话生成模型"),
    tts_model: str = Form(DEFAULT_TTS_MODEL, description="语音合成模型"),
    stt_chat_platform: str = Form("Google", description="STT/Chat 平台：Google / OpenAI"),
    stt_chat_api_url: str = Form(None, description="STT/Chat API 地址"),
    stt_chat_api_key: str = Form(None, description="STT/Chat API Key"),
    stt_chat_model: str = Form(None, description="STT/Chat 模型名称"),
):
    """
    完整的语音对话流程：
    1. STT: 将用户音频转为文字（使用Gemini或OpenAI）
    2. Chat: 发送文字给AI获取回复（使用Gemini或OpenAI）
    3. TTS: 将AI回复转为语音（使用Gemini TTS 或 Minimax T2A）
    
    返回: 包含识别文本、AI回复文本、语音数据的JSON响应
    """
    if not _GENAI_AVAILABLE and stt_chat_platform == "Google":
        raise HTTPException(
            status_code=501,
            detail="google-genai SDK 不可用，请在后端安装：pip install google-genai"
        )
    
    # 兜底处理：如果前端传空字符串，使用默认值
    stt_model = stt_model or DEFAULT_STT_MODEL
    chat_model = chat_model or DEFAULT_CHAT_MODEL
    tts_model = tts_model or DEFAULT_TTS_MODEL
    
    # 如果是 OpenAI 平台，使用前端传来的模型，如果没传则使用默认
    if stt_chat_platform == "OpenAI":
        chat_model = stt_chat_model or "gpt-4o-mini"
        stt_model = "whisper-1" # OpenAI STT 固定用 whisper-1
    elif stt_chat_platform == "Google":
        # Google 平台优先使用 stt_chat_model
        if stt_chat_model:
            chat_model = stt_chat_model
            stt_model = stt_chat_model # Google STT/Chat 通常用同一个模型

    # 调试日志：检查 Key 是否存在 (只打印前4位)
    masked_api_key = f"{api_key[:4]}..." if api_key else "None"
    masked_stt_key = f"{stt_chat_api_key[:4]}..." if stt_chat_api_key else "None"
    logger.info(f"Voice Chat Request - Platform: {stt_chat_platform}, Chat Model: {chat_model}, Voice: {voice_name}")
    logger.info(f"Keys - api_key: {masked_api_key}, stt_chat_api_key: {masked_stt_key}")

    try:
        # 读取用户上传的音频
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="音频数据为空")
        
        user_text = ""
        assistant_text = ""
        
        # ========== 步骤1 & 2: STT & Chat ==========
        if stt_chat_platform == "OpenAI":
            import httpx
            
            if not stt_chat_api_key:
                raise HTTPException(status_code=400, detail="OpenAI API Key 未提供")
                
            base_url = stt_chat_api_url or "https://api.openai.com/v1"
            headers = {"Authorization": f"Bearer {stt_chat_api_key}"}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 1. STT (Whisper)
                logger.info(f"Step 1: Speech-to-Text using OpenAI Whisper")
                
                # OpenAI 需要文件名
                files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
                data = {'model': stt_model}
                
                try:
                    stt_resp = await client.post(f"{base_url}/audio/transcriptions", headers=headers, files=files, data=data)
                    if stt_resp.status_code != 200:
                        logger.error(f"OpenAI STT failed: {stt_resp.text}")
                        raise Exception(f"OpenAI STT failed: {stt_resp.status_code}")
                    
                    user_text = stt_resp.json().get("text", "").strip()
                    logger.info(f"STT Result: {user_text[:100]}...")
                except Exception as e:
                    logger.exception("OpenAI STT error")
                    raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")

                if not user_text:
                    raise HTTPException(status_code=400, detail="无法识别音频内容")

                # 2. Chat (GPT)
                logger.info(f"Step 2: Generate response using OpenAI {chat_model}")
                
                # 构建消息列表
                import json
                try:
                    history = json.loads(chat_history)
                except:
                    history = []
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                # 只保留最近几轮
                recent_history = history[-6:] if len(history) > 6 else history
                for msg in recent_history:
                    messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                
                messages.append({"role": "user", "content": user_text})
                
                chat_payload = {
                    "model": chat_model,
                    "messages": messages,
                    "max_tokens": 150
                }
                
                try:
                    chat_resp = await client.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {stt_chat_api_key}", "Content-Type": "application/json"}, json=chat_payload)
                    if chat_resp.status_code != 200:
                        logger.error(f"OpenAI Chat failed: {chat_resp.text}")
                        raise Exception(f"OpenAI Chat failed: {chat_resp.status_code}")
                        
                    assistant_text = chat_resp.json()['choices'][0]['message']['content'].strip()
                    logger.info(f"Chat Response: {assistant_text[:100]}...")
                except Exception as e:
                    logger.exception("OpenAI Chat error")
                    raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")

        else: # Google (Gemini)
            # 使用传入的 stt_chat_api_key，如果没传则使用原来的 api_key (兼容旧逻辑)
            gemini_key = stt_chat_api_key or api_key
            if not gemini_key:
                 raise HTTPException(status_code=400, detail="Google API Key 未提供")

            client = genai.Client(api_key=gemini_key)
            
            # 1. STT
            logger.info(f"Step 1: Speech-to-Text using Gemini {stt_model}")
            stt_response = client.models.generate_content(
                model=stt_model,
                contents=[
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=audio.content_type or "audio/wav"
                    ),
                    "转录："
                ]
            )
            user_text = stt_response.text.strip()
            logger.info(f"STT Result: {user_text[:100]}...")
            
            if not user_text:
                raise HTTPException(status_code=400, detail="无法识别音频内容")
            
            # 2. Chat
            logger.info(f"Step 2: Generate response using Gemini {chat_model}")
            
            import json
            try:
                history = json.loads(chat_history)
            except:
                history = []
            
            recent_history = history[-6:] if len(history) > 6 else history
            
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(system_prompt)
            
            for msg in recent_history:
                prompt_parts.append(msg.get("content", ""))
            
            prompt_parts.append(user_text)
            full_prompt = "\n".join(prompt_parts)
            
            chat_response = client.models.generate_content(
                model=chat_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=150
                )
            )
            
            if chat_response.text:
                assistant_text = chat_response.text.strip()
                logger.info(f"Chat Response: {assistant_text[:100]}...")
            else:
                logger.warning("Chat Response text is None (possibly blocked or empty)")
                assistant_text = ""

        if not assistant_text:
            assistant_text = "抱歉，我无法理解您的问题。"
        
        # ========== 步骤3: TTS - 语音合成 ==========
        logger.info(f"Step 3: Text-to-Speech using {voice_platform} (Voice: {voice_name})")
        
        audio_base64 = ""
        sample_rate = 24000
        
        try:
            if voice_platform == "Minimax":
                # 使用 Minimax T2A
                # 注意：这里的 api_key 是前端传来的，如果是 Minimax 平台，它就是 Minimax Key
                wav_data, sr = await synthesize_minimax_t2a(assistant_text, voice_name, provider_api_url, api_key)
                if wav_data:
                    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                    sample_rate = sr
            else:
                # 默认使用 Gemini TTS (需要 Google Key)
                # 注意：这里始终使用 Google Key，即使 STT/Chat 用的是 OpenAI
                # 如果 stt_chat_platform 是 Google，直接复用 client
                # 如果是 OpenAI，需要重新初始化 Gemini client (使用 api_key 参数，这是专门传给 TTS 用的)
                
                tts_client = None
                if stt_chat_platform == "Google":
                     tts_client = client
                else:
                     # 如果 STT/Chat 是 OpenAI，TTS 仍需 Gemini，则必须提供 api_key
                     if not api_key:
                         logger.warning("Gemini TTS requires Google API Key, but none provided (stt_chat_platform is OpenAI)")
                     else:
                         tts_client = genai.Client(api_key=api_key)

                if tts_client:
                    # 如果回复太长，截断避免TTS超时（保留前200字符）
                    text_for_tts = assistant_text[:200] if len(assistant_text) > 200 else assistant_text
                    if len(assistant_text) > 200:
                        logger.warning(f"AI response too long ({len(assistant_text)} chars), truncated to 200 for TTS")
                    
                    # 直接传递文本，不添加额外的prompt（加快TTS速度）
                    tts_response = tts_client.models.generate_content(
                        model=tts_model,
                        contents=text_for_tts,  # 直接传递，不需要"请说："
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
                    audio_data = tts_response.candidates[0].content.parts[0].inline_data.data
                    
                    # 转换为WAV格式（更通用）
                    if audio_data:
                        wav_data = wave_file_bytes(audio_data)
                        audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                else:
                     logger.error("Skipping Gemini TTS: No Google API Key available")
                    
        except Exception as tts_error:
            # TTS失败时仍返回文字（配额用尽或其他错误）
            logger.error(f"TTS failed ({voice_platform}): {tts_error}, returning text-only response")
            # 返回空音频，客户端只显示文字
            audio_base64 = ""
        
        # ========== 返回完整结果 ==========
        result = {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "audio_base64": audio_base64,
            "audio_format": "wav",
            "sample_rate": sample_rate,
            "tts_available": bool(audio_base64)  # 标记TTS是否可用
        }
        
        # ========== 返回完整结果 ==========
        result = {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "audio_base64": audio_base64,
            "audio_format": "wav",
            "sample_rate": sample_rate,
            "tts_available": bool(audio_base64)  # 标记TTS是否可用
        }
        
        # 使用orjson序列化（更快）
        import orjson
        return StreamingResponse(
            iter([orjson.dumps(result)]),
            media_type="application/json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Voice chat processing error")
        raise HTTPException(status_code=500, detail=f"语音对话处理失败: {str(e)}")


@router.post("/gemini/voice-chat/stt", summary="语音识别（STT）", tags=["Gemini Voice Chat"])
async def speech_to_text(
    audio: UploadFile = File(..., description="音频文件"),
    api_key: str = Form(..., description="Google AI Studio API Key"),
    model: str = Form(DEFAULT_STT_MODEL, description="识别模型")
):
    """
    使用Gemini的音频理解能力进行语音识别
    """
    if not _GENAI_AVAILABLE:
        raise HTTPException(status_code=501, detail="google-genai SDK 不可用")
    
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="音频数据为空")
        
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=audio.content_type or "audio/wav"
                ),
                "请将这段音频转录为文字。只输出转录的文字内容，不要添加任何其他说明。"
            ]
        )
        
        text = response.text.strip()
        
        return {"text": text}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("STT error")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


@router.post("/gemini/voice-chat/tts", summary="语音合成（TTS）", tags=["Gemini Voice Chat"])
async def text_to_speech(
    text: str = Form(..., description="要合成的文字"),
    api_key: str = Form(..., description="Google AI Studio API Key"),
    voice_name: str = Form(DEFAULT_VOICE_NAME, description="语音名称"),
    model: str = Form(DEFAULT_TTS_MODEL, description="TTS模型"),
    format: str = Form("wav", description="输出格式：wav或pcm")
):
    """
    使用Gemini TTS将文字转换为语音
    
    支持的语音名称（30种）：
    - Kore (坚定), Puck (欢快), Zephyr (明亮), Charon (知性)
    - 更多请参考: https://ai.google.dev/gemini-api/docs/speech-generation#voices
    """
    if not _GENAI_AVAILABLE:
        raise HTTPException(status_code=501, detail="google-genai SDK 不可用")
    
    try:
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=f"请用自然的语气说：{text}",
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
        pcm_data = response.candidates[0].content.parts[0].inline_data.data
        
        if format.lower() == "wav":
            # 转换为WAV格式
            audio_data = wave_file_bytes(pcm_data)
            media_type = "audio/wav"
        else:
            # 返回原始PCM
            audio_data = pcm_data
            media_type = "audio/pcm;rate=24000"
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="speech.{format}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

