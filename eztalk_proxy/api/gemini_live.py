import asyncio
import io
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from eztalk_proxy.services.requests.prompts import compose_voice_system_prompt
from fastapi.responses import StreamingResponse

logger = logging.getLogger("EzTalkProxy.Routers.GeminiLive")
router = APIRouter()

# SDK 兼容处理：优先采用新 SDK（google.genai），如不可用则提示
try:
    # New SDK (supports aio.live)
    from google import genai as google_genai_new  # type: ignore
    from google.genai import types as google_genai_types  # type: ignore
    _NEW_GENAI_AVAILABLE = True
except Exception:
    _NEW_GENAI_AVAILABLE = False

DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

@router.post("/gemini/live/relay-pcm", summary="Gemini Live: 单次PCM转发（服务器到服务器）", tags=["Gemini Live"])
async def relay_pcm_to_gemini_live(
    audio: UploadFile = File(..., description="16-bit PCM, 16kHz, mono"),
    api_key: str = Form(..., description="Google AI Studio API Key，仅用于本次请求，不会持久化"),
    model: str = Form(DEFAULT_MODEL, description="Live 模型，默认 gemini-2.5-flash-native-audio-preview-09-2025")
):
    """
    服务器到服务器简化版：接受客户端上传的一段PCM(16k)音频，将其通过 Live API 发送给 Gemini，
    并把 Gemini 返回的音频（24k PCM）以流的形式回传给客户端。

    说明：
    - 本端返回 Content-Type: audio/pcm;rate=24000，便于前端直接播放/拼接。
    - 若需要连续会话/双向全双工，请在后续迭代中切换为真正的WebSocket双工转发（计划新建 /gemini/live/ws）。
    - 当前端语音模式选择“Gemini”且用户已在设置中填写Key时，直接用该接口即可打通最小可用链路。
    """
    if not _NEW_GENAI_AVAILABLE:
        # 新SDK不可用，给出明确提示
        raise HTTPException(
            status_code=501,
            detail="google.genai SDK 不可用，请在后端安装/升级新版 Google GenAI SDK：pip install google-genai"
        )

    if audio.content_type not in ("application/octet-stream", "audio/pcm", "audio/x-pcm", "audio/wav", "audio/x-wav"):
        logger.warning(f"Unexpected content_type for PCM input: {audio.content_type}")
        # 不强制失败，尽量兼容

    try:
        # 读取整段PCM（16k，16-bit，mono）
        pcm_bytes = await audio.read()
        if not pcm_bytes:
            raise HTTPException(status_code=400, detail="空的音频数据")

        client = google_genai_new.Client(api_key=api_key)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("初始化 Google GenAI 客户端失败")
        raise HTTPException(status_code=500, detail=f"初始化 Google GenAI 客户端失败: {e}")

    async def stream_out():
        """
        将 Live API 的 24k PCM 回传给客户端（Content-Type: audio/pcm;rate=24000）。
        """
        try:
            system_instruction = compose_voice_system_prompt()
            cfg = {
                "response_modalities": ["AUDIO"],
                "system_instruction": system_instruction,
            }
            # 建立 Live 会话
            async with client.aio.live.connect(model=model, config=cfg) as session:
                # 发送输入（16k PCM）
                await session.send_realtime_input(
                    audio=google_genai_types.Blob(
                        data=pcm_bytes,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

                # 持续接收 24k PCM 输出并向下游发送
                async for resp in session.receive():
                    # resp.data 即音频字节（24k 采样）
                    if resp.data:
                        # 直接向客户端输出裸 PCM
                        yield resp.data
                        # 协程让出，避免阻塞事件循环
                        await asyncio.sleep(0)

        except Exception as e:
            logger.exception("Gemini Live 转发过程中发生错误")
            # 将错误转为可读字节，避免中断连接（客户端可选择忽略尾部错误）
            err_msg = f"[Live Error] {str(e)}".encode("utf-8", errors="ignore")
            # 发送一小段静默或错误信息（根据需要可以去掉）
            yield err_msg

    # 返回 24k 裸 PCM 音频流
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Content-Disposition": 'inline; filename="gemini_live_24k.pcm"'
    }
    return StreamingResponse(stream_out(), media_type="audio/pcm;rate=24000", headers=headers)