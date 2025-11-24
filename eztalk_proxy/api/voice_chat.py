import logging
import base64
import io
import wave
import json
import orjson
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

# 导入新的处理器
from ..services.voice import google_handler, openai_handler, minimax_handler, siliconflow_handler
from eztalk_proxy.services.requests.prompts import compose_voice_system_prompt

logger = logging.getLogger("EzTalkProxy.Routers.VoiceChat")
router = APIRouter()

DEFAULT_VOICE_NAME = "Kore"

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

@router.post("/complete", summary="完整语音对话：STT → Chat → TTS", tags=["Voice Chat"])
async def complete_voice_chat(
    audio: UploadFile = File(..., description="用户录音文件"),
    
    # --- 1. STT 配置 ---
    stt_platform: str = Form("Google", description="STT 平台: Google / OpenAI"),
    stt_api_key: str = Form(None),
    stt_api_url: str = Form(None),
    stt_model: str = Form(None),
    
    # --- 2. Chat (LLM) 配置 ---
    chat_platform: str = Form("Google", description="Chat 平台: Google / OpenAI"),
    chat_api_key: str = Form(None),
    chat_api_url: str = Form(None),
    chat_model: str = Form(None),
    chat_history: str = Form("[]"),
    system_prompt: str = Form(""),
    
    # --- 3. TTS 配置 ---
    voice_platform: str = Form("Gemini", description="TTS 平台: Gemini / Minimax"),
    voice_name: str = Form(DEFAULT_VOICE_NAME),
    tts_api_key: str = Form(None),
    tts_api_url: str = Form(None), # Minimax 需要
    tts_model: str = Form(None),
    
    # --- 兼容旧参数 (Legacy) ---
    api_key: str = Form(None), # 旧版 Google Key
    stt_chat_platform: str = Form(None), # 旧版 STT/Chat 混合平台
    stt_chat_api_key: str = Form(None),
    stt_chat_api_url: str = Form(None),
    stt_chat_model: str = Form(None),
    provider_api_url: str = Form(None), # 旧版 Minimax URL
):
    """
    完整的语音对话流程：STT -> Chat -> TTS
    支持多厂商混合调用
    """
    # === 参数归一化处理 ===
    
    # 1. STT 参数解析
    final_stt_platform = stt_platform
    if stt_chat_platform and stt_platform == "Google":
        final_stt_platform = stt_chat_platform
        
    final_stt_key = stt_api_key or stt_chat_api_key or api_key
    final_stt_url = stt_api_url or stt_chat_api_url
    final_stt_model = stt_model or stt_chat_model
    
    if not final_stt_model:
        raise HTTPException(status_code=400, detail="STT 模型名称未填写")
    
    # SiliconFlow 也需要 API 地址（虽然前端会固定传入，但后端校验逻辑保持一致）
    if final_stt_platform != "Google" and not final_stt_url:
        raise HTTPException(status_code=400, detail=f"{final_stt_platform} STT API 地址未填写")

    # 2. Chat 参数解析
    final_chat_platform = chat_platform
    if stt_chat_platform and chat_platform == "Google":
        final_chat_platform = stt_chat_platform
        
    final_chat_key = chat_api_key or stt_chat_api_key or api_key
    final_chat_url = chat_api_url or stt_chat_api_url
    final_chat_model = chat_model or stt_chat_model
    
    if not final_chat_model:
        raise HTTPException(status_code=400, detail="Chat 模型名称未填写")
    if final_chat_platform != "Google" and not final_chat_url:
        raise HTTPException(status_code=400, detail=f"{final_chat_platform} Chat API 地址未填写")

    # 3. TTS 参数解析
    final_tts_platform = voice_platform
    final_tts_key = tts_api_key or api_key
    final_tts_url = tts_api_url or provider_api_url
    
    if final_tts_platform == "Minimax" and not final_tts_key:
        final_tts_key = api_key
    
    # TTS 模型必填校验
    # 注意：Gemini TTS 的模型名参数 tts_model 可能为空时需要校验
    if not tts_model:
         raise HTTPException(status_code=400, detail="TTS 模型名称未填写")
    
    if final_tts_platform == "Minimax" and not final_tts_url:
        raise HTTPException(status_code=400, detail="Minimax TTS API 地址未填写")

    logger.info(f"Voice Chat Request:")
    logger.info(f"  STT: {final_stt_platform} ({final_stt_model})")
    logger.info(f"  Chat: {final_chat_platform} ({final_chat_model})")
    logger.info(f"  TTS: {final_tts_platform} ({voice_name})")

    try:
        # 读取音频
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="音频数据为空")
        
        user_text = ""
        assistant_text = ""
        
        # ========== Step 1: STT ==========
        if final_stt_platform == "OpenAI":
            user_text = await openai_handler.process_stt(
                audio_bytes=audio_bytes,
                api_key=final_stt_key,
                api_url=final_stt_url,
                model=final_stt_model
            )
        elif final_stt_platform == "SiliconFlow":
            user_text = await siliconflow_handler.process_stt(
                audio_bytes=audio_bytes,
                api_key=final_stt_key,
                api_url=final_stt_url,
                model=final_stt_model,
                mime_type=audio.content_type or "audio/wav"
            )
        else: # Google
            user_text = google_handler.process_stt(
                audio_bytes=audio_bytes,
                mime_type=audio.content_type or "audio/wav",
                api_key=final_stt_key,
                model=final_stt_model,
                api_url=final_stt_url
            )
            
        if not user_text:
            raise HTTPException(status_code=400, detail="无法识别音频内容")
            
        # ========== Step 2: Chat ==========
        # 解析历史记录
        try:
            history_list = json.loads(chat_history)
        except:
            history_list = []
            
        # 优化语音模式 Prompt
        voice_prompt = compose_voice_system_prompt()
        if system_prompt:
            final_system_prompt = f"{voice_prompt}\n\n[补充要求]\n{system_prompt}"
        else:
            final_system_prompt = voice_prompt

        if final_chat_platform == "OpenAI":
            assistant_text = await openai_handler.process_chat(
                user_text=user_text,
                chat_history=history_list,
                system_prompt=final_system_prompt,
                api_key=final_chat_key,
                api_url=final_chat_url,
                model=final_chat_model
            )
        else: # Google
            assistant_text = google_handler.process_chat(
                user_text=user_text,
                chat_history=history_list,
                system_prompt=final_system_prompt,
                api_key=final_chat_key,
                model=final_chat_model,
                api_url=final_chat_url
            )
            
        if not assistant_text:
            assistant_text = "抱歉，我无法理解您的问题。"
            
        # ========== Step 3: TTS ==========
        audio_base64 = ""
        sample_rate = 24000
        
        try:
            if final_tts_platform == "Minimax":
                wav_data, sr = await minimax_handler.synthesize_minimax_t2a(
                    text=assistant_text,
                    voice_id=voice_name,
                    api_url=final_tts_url,
                    api_key=final_tts_key
                )
                if wav_data:
                    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                    sample_rate = sr
            else: # Gemini
                # Gemini TTS 需要 Google Key
                # 如果 Chat 用了 OpenAI，这里 final_tts_key 可能为空，需要检查
                if not final_tts_key:
                     logger.warning("Gemini TTS requires Google API Key, but none provided")
                else:
                    pcm_data = google_handler.process_tts(
                        text=assistant_text,
                        api_key=final_tts_key,
                        voice_name=voice_name,
                        model=tts_model,
                        api_url=final_tts_url
                    )
                    if pcm_data:
                        wav_data = wave_file_bytes(pcm_data)
                        audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                        
        except Exception as tts_e:
            logger.error(f"TTS failed: {tts_e}")
            # TTS 失败不阻断流程，只返回文字
            
        # ========== 返回结果 ==========
        result = {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "audio_base64": audio_base64,
            "audio_format": "wav",
            "sample_rate": sample_rate,
            "tts_available": bool(audio_base64)
        }
        
        return StreamingResponse(
            iter([orjson.dumps(result)]),
            media_type="application/json"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Voice chat processing error")
        raise HTTPException(status_code=500, detail=f"语音对话处理失败: {str(e)}")

# 保留单独的 STT 和 TTS 接口供测试或其他用途，也可以重构为使用 handler
@router.post("/stt", summary="语音识别（STT）", tags=["Voice Chat"])
async def speech_to_text(
    audio: UploadFile = File(...),
    api_key: str = Form(...),
    model: str = Form(...),
    platform: str = Form("Google"),
    api_url: str = Form(None)
):
    audio_bytes = await audio.read()
    mime = audio.content_type or "audio/wav"
    
    if platform == "OpenAI":
        if not api_url:
            raise HTTPException(400, "OpenAI STT 需要提供 api_url")
        text = await openai_handler.process_stt(audio_bytes, api_key, api_url, model)
    elif platform == "SiliconFlow":
        if not api_url:
            raise HTTPException(400, "SiliconFlow STT 需要提供 api_url")
        text = await siliconflow_handler.process_stt(audio_bytes, api_key, api_url, model, mime)
    else:
        # 默认 Google
        text = google_handler.process_stt(audio_bytes, mime, api_key, model, api_url)
        
    return {"text": text}

@router.post("/tts", summary="语音合成（TTS）", tags=["Voice Chat"])
async def text_to_speech(
    text: str = Form(...),
    api_key: str = Form(...),
    voice_name: str = Form(DEFAULT_VOICE_NAME),
    model: str = Form(...),
    format: str = Form("wav")
):
    # 默认使用 Google
    pcm_data = google_handler.process_tts(text, api_key, voice_name, model)
    
    if format.lower() == "wav":
        audio_data = wave_file_bytes(pcm_data)
        media_type = "audio/wav"
    else:
        audio_data = pcm_data
        media_type = "audio/pcm;rate=24000"
        
    return StreamingResponse(
        io.BytesIO(audio_data),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="speech.{format}"'}
    )