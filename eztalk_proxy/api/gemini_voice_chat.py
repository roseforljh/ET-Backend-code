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

DEFAULT_STT_MODEL = "gemini-2.5-flash"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE_NAME = "Kore"  # 可选: Zephyr, Puck, Kore, 等30种


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


@router.post("/gemini/voice-chat/complete", summary="完整语音对话：STT → Chat → TTS", tags=["Gemini Voice Chat"])
async def complete_voice_chat(
    audio: UploadFile = File(..., description="用户录音文件（支持多种格式：wav, mp3, ogg, flac等）"),
    api_key: str = Form(..., description="Google AI Studio API Key"),
    chat_history: str = Form("[]", description="对话历史，JSON数组格式，例如：[{\"role\":\"user\",\"content\":\"你好\"},{\"role\":\"assistant\",\"content\":\"你好！\"}]"),
    system_prompt: str = Form("", description="系统提示词（可选）"),
    voice_name: str = Form(DEFAULT_VOICE_NAME, description="TTS语音名称（可选）"),
    stt_model: str = Form(DEFAULT_STT_MODEL, description="语音识别模型"),
    tts_model: str = Form(DEFAULT_TTS_MODEL, description="语音合成模型"),
):
    """
    完整的语音对话流程：
    1. STT: 将用户音频转为文字（使用Gemini的音频理解能力）
    2. Chat: 发送文字给Gemini获取回复
    3. TTS: 将Gemini回复转为语音（使用Gemini TTS）
    
    返回: 包含识别文本、AI回复文本、语音数据的JSON响应
    """
    if not _GENAI_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="google-genai SDK 不可用，请在后端安装：pip install google-genai"
        )
    
    try:
        # 读取用户上传的音频
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="音频数据为空")
        
        client = genai.Client(api_key=api_key)
        
        # ========== 步骤1: STT - 语音识别 ==========
        logger.info(f"Step 1: Speech-to-Text using {stt_model}")
        
        # 直接使用音频字节数据
        stt_response = client.models.generate_content(
            model=stt_model,
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=audio.content_type or "audio/wav"
                ),
                "请将这段音频转录为文字。只输出转录的文字内容，不要添加任何其他说明。"
            ]
        )
        
        user_text = stt_response.text.strip()
        logger.info(f"STT Result: {user_text[:100]}...")
        
        if not user_text:
            raise HTTPException(status_code=400, detail="无法识别音频内容，请重试")
        
        # ========== 步骤2: Chat - 对话生成 ==========
        logger.info(f"Step 2: Generate response")
        
        # 构建对话历史
        import json
        try:
            history = json.loads(chat_history)
        except:
            history = []
        
        # 构建完整的对话上下文
        conversation_parts = []
        
        # 添加系统提示词（如果有）
        if system_prompt:
            conversation_parts.append(f"系统提示：{system_prompt}\n\n")
        
        # 添加历史对话
        if history:
            conversation_parts.append("对话历史：\n")
            for msg in history[-10:]:  # 只保留最近10轮对话
                role = "用户" if msg.get("role") == "user" else "助手"
                content = msg.get("content", "")
                conversation_parts.append(f"{role}: {content}\n")
            conversation_parts.append("\n")
        
        # 添加当前用户输入
        conversation_parts.append(f"用户: {user_text}\n助手: ")
        
        full_prompt = "".join(conversation_parts)
        
        chat_response = client.models.generate_content(
            model=stt_model,
            contents=full_prompt
        )
        
        assistant_text = chat_response.text.strip()
        logger.info(f"Chat Response: {assistant_text[:100]}...")
        
        if not assistant_text:
            assistant_text = "抱歉，我无法理解您的问题。"
        
        # ========== 步骤3: TTS - 语音合成 ==========
        logger.info(f"Step 3: Text-to-Speech using {tts_model} with voice {voice_name}")
        
        tts_response = client.models.generate_content(
            model=tts_model,
            contents=f"请用自然的语气说：{assistant_text}",
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
        wav_data = wave_file_bytes(audio_data)
        
        # 将WAV编码为base64以便在JSON中传输
        audio_base64 = base64.b64encode(wav_data).decode('utf-8')
        
        # ========== 返回完整结果 ==========
        result = {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "audio_base64": audio_base64,
            "audio_format": "wav",
            "sample_rate": 24000
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

