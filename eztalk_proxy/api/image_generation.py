from fastapi import APIRouter, HTTPException, Body, Request
from fastapi import Response
from ..models.image_generation_api_models import ImageGenerationRequest, ImageGenerationResponse, ImageUrl
import httpx
import logging
import random
import re
import asyncio
import time
import hashlib
from collections import deque
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from ..services.image_store import save_images, list_images
# Default image provider (SiliconFlow) presets
from ..core.config import (
    SILICONFLOW_IMAGE_API_URL,
    SILICONFLOW_DEFAULT_IMAGE_MODEL,
    SILICONFLOW_API_KEY_DEFAULT,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ========== In-memory session history (text-only) for Gemini native ==========
# Key: hash(client_ip + model + apiAddress), Value: deque of turns [{"role": "user"|"model", "text": str, "ts": float}]
_SESSION_HISTORY: Dict[str, deque] = {}
_SESSION_META: Dict[str, float] = {}  # last access timestamp
# Also store last generated image (data URI) for reference continuity
_SESSION_LAST_IMAGE: Dict[str, str] = {}
# Store upstream "turn token" to continue native Gemini conversation (when provided by upstream)
_SESSION_TURN_TOKEN: Dict[str, str] = {}
# Limits (disabled as per requirements: no TTL, no per-session limits)
_TURN_MAX = None            # keep all turns (no limit)
_TRUNCATE_PER_TURN = None   # do not truncate per turn
_TOTAL_MAX = None           # no total budget
_TTL_SECONDS = None         # TTL disabled

# ========== Force Data-URI (Base64) helpers ==========
import base64

async def _force_images_to_data_uri(normalized: ImageGenerationResponse) -> ImageGenerationResponse:
    """
    Ensure all images in response are Data URI (Base64).
    - If an image url is http(s), fetch bytes and convert to data:{mime};base64,{...}
    - If already data URI, keep as-is.
    On failure to fetch, preserves original URL to avoid breaking responses.
    """
    try:
        images = getattr(normalized, "images", None)
        if not images:
            return normalized

        new_images: list[ImageUrl] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), http2=True, follow_redirects=True) as client:
            for img in images:
                url_val = getattr(img, "url", None)
                if isinstance(url_val, str) and url_val.startswith(("http://", "https://")):
                    try:
                        resp = await client.get(url_val)
                        resp.raise_for_status()
                        content_type = resp.headers.get("Content-Type", "image/png")
                        b64 = base64.b64encode(resp.content).decode("utf-8")
                        data_uri = f"data:{content_type};base64,{b64}"
                        new_images.append(ImageUrl(url=data_uri))
                    except Exception as fetch_err:
                        logger.warning(f"[IMG] Failed to fetch image for Data URI conversion: {fetch_err}")
                        new_images.append(ImageUrl(url=url_val))
                else:
                    # Already data URI or unsupported scheme -> keep
                    new_images.append(ImageUrl(url=url_val))

        return ImageGenerationResponse(
            images=new_images,
            text=normalized.text,
            timings=normalized.timings,
            seed=normalized.seed
        )
    except Exception as e:
        logger.error(f"[IMG] _force_images_to_data_uri failed: {e}", exc_info=True)
        return normalized


def _gc_sessions(now_ts: float) -> None:
    try:
        expired = [k for k, ts in _SESSION_META.items() if now_ts - ts > _TTL_SECONDS]
        for k in expired:
            _SESSION_META.pop(k, None)
            _SESSION_HISTORY.pop(k, None)
            _SESSION_LAST_IMAGE.pop(k, None)
            _SESSION_TURN_TOKEN.pop(k, None)
    except Exception:
        pass

def _make_session_key(client_ip: str, model: str, api_address: str) -> str:
    seed = f"{client_ip}|{model}|{api_address}".encode("utf-8", errors="ignore")
    return hashlib.sha1(seed).hexdigest()

def _truncate_text(s: str, limit: Optional[int]) -> str:
    if not isinstance(s, str):
        return ""
    if limit is None:
        return s
    if len(s) <= limit:
        return s
    # keep head and tail for better semantics
    keep_head = int(limit * 0.6)
    keep_tail = limit - keep_head
    return s[:keep_head] + " ... " + s[-keep_tail:]

def _get_history_for_contents(session_key: str) -> List[Dict[str, Any]]:
    turns = _SESSION_HISTORY.get(session_key)
    if not turns:
        return []
    # apply optional per-turn truncate and optional total budget
    result: List[Dict[str, Any]] = []
    total = 0
    src = list(turns)
    # If _TURN_MAX is set, only take last N; otherwise take all
    if isinstance(_TURN_MAX, int) and _TURN_MAX > 0:
        src = src[-_TURN_MAX:]
    for t in src:
        text = _truncate_text(t.get("text") or "", _TRUNCATE_PER_TURN)
        if not text:
            continue
        if isinstance(_TOTAL_MAX, int) and _TOTAL_MAX > 0 and (total + len(text) > _TOTAL_MAX):
            break
        role = t.get("role")  # "user" or "model"
        result.append({"role": "user" if role == "user" else "model", "parts": [{"text": text}]})
        total += len(text)
    return result

def _append_turn(session_key: str, role: str, text: str) -> None:
    if not text:
        return
    dq = _SESSION_HISTORY.get(session_key)
    if dq is None:
        dq = deque(maxlen=_TURN_MAX)
        _SESSION_HISTORY[session_key] = dq
    dq.append({"role": "user" if role == "user" else "model", "text": _truncate_text(text, _TRUNCATE_PER_TURN), "ts": time.time()})
    _SESSION_META[session_key] = time.time()

def _is_google_official_api(api_address: str) -> bool:
    """Check if the API address is Google's official Gemini API"""
    if not api_address:
        return False
    
    google_domains = [
        "generativelanguage.googleapis.com",
        "ai.google.dev", 
        "googleapis.com"
    ]
    
    api_address_lower = api_address.lower()
    return any(domain in api_address_lower for domain in google_domains)

def _fallback_response(reason: str, user_text: str = None) -> ImageGenerationResponse:
    # 统一的兜底结构，避免前端反序列化报缺少必填字段
    logger.error(f"[IMG] Fallback response due to error: {reason}")
    return ImageGenerationResponse(
        images=[],
        text=user_text,
        timings={"inference": 0},
        seed=random.randint(1, 2**31 - 1)
    )

def _as_image_urls(ext_images: Any) -> List[Dict[str, str]]:
    urls: List[Dict[str, str]] = []
    if not isinstance(ext_images, list):
        return urls

    for item in ext_images:
        if isinstance(item, str) and item.startswith(('http://', 'https://', 'data:image/')):
            urls.append({"url": item})
        elif isinstance(item, dict):
            if "url" in item and isinstance(item["url"], str):
                urls.append({"url": item["url"]})
            elif "b64_json" in item and isinstance(item["b64_json"], str):
                urls.append({"url": f"data:image/png;base64,{item['b64_json']}"})
            # 兼容一些API将b64字符串直接放在image字段的情况
            elif "image" in item and isinstance(item["image"], str):
                urls.append({"url": f"data:image/png;base64,{item['image']}"})
            # 兼容一些API将b64字符串放在更深层嵌套的情况
            elif "image" in item and isinstance(item.get("image"), dict) and isinstance(item["image"].get("b64_json"), str):
                urls.append({"url": f"data:image/png;base64,{item['image']['b64_json']}"})
    return urls

def _normalize_response(data: Dict[str, Any], append_failure_hint: bool = False) -> ImageGenerationResponse:
    images_list: List[Dict[str, str]] = []
    text_parts: List[str] = []

    # Case 1: Provider wraps Gemini image response in an OpenAI chat completion format.
    if "choices" in data and isinstance(data.get("choices"), list) and data["choices"]:
        choice = data["choices"][0]
        if choice.get("finish_reason") == "content_filter":
            return ImageGenerationResponse(
                images=[],
                text="[CONTENT_FILTER]您的请求可能违反了相关的内容安全策略，已被拦截。请修改您的提示后重试。",
                timings={"inference": 0},
                seed=random.randint(1, 2**31 - 1)
            )

        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            # Regex to find markdown image syntax with data URI or standard URL
            # ![...](...)
            url_matches = re.findall(r'!\[.*?\]\((data:image/[^;]+;base64,[^\s\)"]+|https?://[^\s\)]+)\)', content)
            for url in url_matches:
                images_list.append({"url": url})
            
            # Clean the image markdown from the text to get remaining text
            text_content = re.sub(r'!\[.*?\]\((data:image/[^;]+;base64,[^\s\)"]+|https?://[^\s\)]+)\)', "", content).strip()
            if text_content and text_content != '`':
                text_parts.append(text_content)
        elif isinstance(content, list):  # Handle list content (e.g. for Gemini Vision / OpenAI-compat multimodal)
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                # 文本
                if ptype == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part.get("text", ""))
                # OpenAI 兼容 image_url
                elif ptype == "image_url":
                    image_url_data = part.get("image_url", {})
                    if isinstance(image_url_data, dict) and "url" in image_url_data:
                        images_list.append({"url": image_url_data["url"]})
                # inline_data / inlineData（常见于Gemini返回或部分中转商）
                elif ptype in ("inline_data", "inlineData"):
                    data_obj = part.get("inline_data") or part.get("inlineData") or {}
                    if isinstance(data_obj, dict) and isinstance(data_obj.get("data"), str):
                        mime = data_obj.get("mime_type") or data_obj.get("mimeType") or "image/png"
                        images_list.append({"url": f"data:{mime};base64,{data_obj['data']}"})
                # image + b64_json/data
                elif ptype in ("image", "image_base64"):
                    img_obj = part.get("image", {})
                    if isinstance(img_obj, dict):
                        if isinstance(img_obj.get("b64_json"), str):
                            images_list.append({"url": f"data:image/png;base64,{img_obj['b64_json']}"})
                        elif isinstance(img_obj.get("data"), str):
                            data_val = img_obj["data"]
                            if data_val.startswith("data:image/"):
                                images_list.append({"url": data_val})
                            else:
                                images_list.append({"url": f"data:image/png;base64,{data_val}"})
                # 一些中转商可能直接把 data URI 放在字段 url 上
                elif isinstance(part.get("url"), str) and part["url"].startswith(("http://", "https://", "data:image/")):
                    images_list.append({"url": part["url"]})

        # Case 1.5: Handle OpenRouter's non-standard format for Gemini Image
        if not images_list and "images" in message and isinstance(message.get("images"), list):
            for img_item in message["images"]:
                if isinstance(img_item, dict) and img_item.get("type") == "image_url":
                    img_url_data = img_item.get("image_url", {})
                    if isinstance(img_url_data, dict) and "url" in img_url_data:
                        images_list.append({"url": img_url_data["url"]})

    # Case 2: Gemini's native format
    elif "candidates" in data and isinstance(data["candidates"], list):
        for candidate in data["candidates"]:
            if isinstance(candidate.get("content"), dict) and isinstance(candidate["content"].get("parts"), list):
                for part in candidate["content"]["parts"]:
                    if isinstance(part.get("inlineData"), dict) and isinstance(part["inlineData"].get("data"), str):
                        images_list.append({"url": f"data:image/png;base64,{part['inlineData']['data']}"})
                    if isinstance(part.get("text"), str):
                        text_parts.append(part["text"])

    # Case 3: Standard DALL-E/SD format (if no images found in other structures)
    if not images_list:
        if "images" in data:
            images_list = _as_image_urls(data.get("images"))
        elif "data" in data:
            images_list = _as_image_urls(data.get("data"))
        elif "output" in data and isinstance(data["output"], dict):
            images_list = _as_image_urls(data["output"].get("images"))
        elif "image" in data:  # Fallback for single image field
            images_list = _as_image_urls([data["image"]] if data["image"] else [])

    # Case 3.1: Additional provider variants (Kolors/Qwen/common aggregators)
    if not images_list and isinstance(data, dict):
        # result.images or result.data (some providers nest under 'result')
        try:
            result_obj = data.get("result")
            if isinstance(result_obj, dict):
                if not images_list and isinstance(result_obj.get("images"), (list, dict, str)):
                    images_list = _as_image_urls(result_obj.get("images"))
                if not images_list and isinstance(result_obj.get("data"), (list, dict, str)):
                    images_list = _as_image_urls(result_obj.get("data"))
        except Exception:
            pass
        # artifacts with base64 (common in SD/stability-like payloads)
        try:
            artifacts = data.get("artifacts")
            if not images_list and isinstance(artifacts, list):
                # Map artifacts -> data URI list
                buf = []
                for a in artifacts:
                    if isinstance(a, dict):
                        b64v = a.get("base64") or a.get("b64_json")
                        urlv = a.get("url")
                        if isinstance(urlv, str):
                            buf.append({"url": urlv})
                        elif isinstance(b64v, str):
                            buf.append({"url": f"data:image/png;base64,{b64v}"})
                if buf:
                    images_list = buf
        except Exception:
            pass
        # images_url / image_url (non-standard)
        try:
            images_url_val = data.get("images_url") or data.get("image_urls")
            if not images_list and images_url_val is not None:
                images_list = _as_image_urls(images_url_val)
            image_url_val = data.get("image_url")
            if not images_list and image_url_val is not None:
                images_list = _as_image_urls([image_url_val] if not isinstance(image_url_val, list) else image_url_val)
        except Exception:
            pass
        # nested output list objects like {"output":[{"url":...},{"b64_json":...}]}
        try:
            output_val = data.get("output")
            if not images_list and isinstance(output_val, list):
                images_list = _as_image_urls(output_val)
        except Exception:
            pass

    # Consolidate text and check for image generation failure patterns
    final_text = " ".join(text_parts).strip()
    if not images_list and final_text:
        # Common phrases indicating an intended but failed image generation
        failure_patterns = [
            "好的，这是", "好的，这是您要的", "这是您要的图片", "生成的图片如下", "这是我为您生成的",
            "here is the image", "here are the images", "i have generated", "voici l'image", "ecco l'immagine"
        ]
        # Use lowercasing for case-insensitive matching
        final_text_lower = final_text.lower()
        if append_failure_hint and any(p.lower() in final_text_lower for p in failure_patterns):
            # Append a user-friendly message about the potential failure
            failure_message = "\n\n(图片生成失败或被模型拒绝。请稍后重试或更换提示词。)"
            final_text += failure_message

    if not images_list and not final_text:
        raise ValueError("Downstream API did not return any recognizable images or text field")

    # Timings and Seed logic remains the same
    timings_obj = {}
    if isinstance(data.get("timings"), dict) and "inference" in data["timings"]:
        timings_obj = {"inference": int(data["timings"]["inference"])}
    else:
        inference_ms = None
        for key in ["inference", "inference_ms", "latency_ms", "runtime_ms"]:
            if isinstance(data.get(key), (int, float)):
                inference_ms = int(data[key])
                break
        timings_obj = {"inference": int(inference_ms or 0)}

    seed_val = data.get("seed")
    if not isinstance(seed_val, int):
        for k in ["meta", "metadata"]:
            maybe = data.get(k, {})
            if isinstance(maybe, dict) and isinstance(maybe.get("seed"), int):
                seed_val = maybe["seed"]
                break
        if not isinstance(seed_val, int):
            seed_val = random.randint(1, 2**31 - 1)

    # 始终返回可序列化的文本字段：若有图片但无文本，则返回空串，避免前端“无文本不保存”的逻辑短路
    normalized = {
        "images": images_list,
        "text": (final_text if final_text else ("")) if images_list else (final_text if final_text else None),
        "timings": timings_obj,
        "seed": seed_val
    }
    return ImageGenerationResponse(**normalized)

async def _proxy_and_normalize(request: ImageGenerationRequest, request_obj: Optional[Request] = None, response_obj: Optional[Response] = None) -> ImageGenerationResponse:
    url = request.apiAddress
    # 清理意外携带的 '#...' 片段，避免无效路径
    try:
        if isinstance(url, str) and "#" in url:
            url = url.split("#", 1)[0]
    except Exception:
        pass

    # 有效参数的本地注入占位（可能被“默认平台”覆盖）
    effective_api_key = request.apiKey
    effective_url = url
    effective_model = request.model

    headers = {
        "Authorization": f"Bearer {effective_api_key or ''}",
        "Content-Type": "application/json",
        "User-Agent": "EzTalkProxy/1.9.9",
        "Accept": "application/json"
    }
    payload = {}
    seedream_mode = False

    model_lower = (effective_model or "").lower()
    is_gemini_image_model = "gemini" in model_lower and ("flash-image" in model_lower or "gemini-pro-vision" in model_lower)
    provider = request.provider or "openai compatible"
    provider_lower = provider.lower()
    
    # 会话键（仅用于 Gemini 原生连续对话）
    # 根本修复：历史项必须严格按 conversationId 隔离。若缺失，则完全禁用服务端会话（不再以IP回退），避免跨历史项串联。
    try:
        conv_id = getattr(request, "conversation_id", None)
    except Exception:
        conv_id = None
    try:
        client_ip = (request_obj.client.host if request_obj and request_obj.client else "unknown")
    except Exception:
        client_ip = "unknown"
    enable_session = bool(conv_id)  # 只有前端显式提供 conversationId 才启用连续会话
    # 使用清洗后的 url 参与键生成，避免同一个地址因为 '#' 或路径后缀差异导致键不一致
    cleaned_api = (url or "") if isinstance(url, str) else (request.apiAddress or "")
    session_key = _make_session_key(conv_id, request.model or "", cleaned_api) if enable_session else None
    # 无会话ID也要保证可持久化：以 client_ip + model + cleaned_api 生成稳定键
    try:
        persist_key = conv_id or _make_session_key(client_ip, request.model or "", cleaned_api)
    except Exception:
        persist_key = conv_id or client_ip or "unknown"
    # 将持久化键通过响应头返回，便于前端固定拉取历史
    try:
        if response_obj is not None and isinstance(persist_key, str) and persist_key:
            response_obj.headers["X-Image-History-Key"] = persist_key
    except Exception:
        pass
    now_ts = time.time()
    # TTL disabled: do not garbage collect sessions based on time
    if enable_session and session_key:
        _SESSION_META[session_key] = now_ts
        try:
            has_cached_ref = isinstance(_SESSION_LAST_IMAGE.get(session_key), str)
            logger.info(f"[IMG DEBUG] Session ON key={session_key[:8]}..., turns={len(_SESSION_HISTORY.get(session_key, []))}, has_ref={has_cached_ref}")
        except Exception:
            pass
    else:
        logger.info(f"[IMG DEBUG] Session OFF (no conversationId). Isolation enforced per-request. client_ip={client_ip}")

    # 调试信息：显示接收到的完整请求信息
    logger.info(f"[IMG DEBUG] Received ImageGenRequest:")
    logger.info(f"  - model: {effective_model}")
    logger.info(f"  - provider/channel: {request.provider}")
    logger.info(f"  - apiAddress: {request.apiAddress}")
    logger.info(f"  - apiKey: {(effective_api_key[:10] + '...') if isinstance(effective_api_key, str) and effective_api_key else 'None'}")
    session_key_for_log = f"{session_key[:8]}..." if session_key else "(none)"
    logger.info(f"[IMG DEBUG] Session key: {session_key_for_log}, client_ip: {client_ip}, conv_id: {conv_id or '(none)'}")

    # 根据用户选择的渠道/提供商决定API格式
    # 旧逻辑仅做精确映射，容易被“Gemini 原生渠道/原生/Gemini渠道”等变体漏判，改为鲁棒归一化：
    def _normalize_channel(p: str) -> str:
        """
        渠道判定规则（严格以“渠道”语义为准，不再依赖平台名称推断）：
        - 明确为“Gemini”（大小写/中英文均可）→ gemini
        - 明确为“OpenAI兼容/openai compatible/compat/兼容” → openai_compatible
        - 其他任意值 → openai_compatible（安全回退）
        """
        try:
            if not isinstance(p, str):
                return "openai_compatible"
            pl = p.strip().lower()
            # 精确优先：等同“gemini”的渠道字符串
            if pl in ("gemini", "谷歌", "google", "gemini渠道", "gemini原生"):
                return "gemini"
            # 广义包含：包含 gemini/google/谷歌/原生 关键字
            if any(k in pl for k in ("gemini", "google", "谷歌", "原生")):
                return "gemini"
            # OpenAI 兼容渠道（中英均可）
            if pl in ("openai兼容", "openai compatible", "compat", "兼容"):
                return "openai_compatible"
            if any(k in pl for k in ("openai", "兼容", "compat")):
                return "openai_compatible"
            # 未知渠道统一走 openai 兼容，避免误判
            return "openai_compatible"
        except Exception:
            return "openai_compatible"

    normalized_channel = _normalize_channel(provider)
    
    logger.info(f"[IMG DEBUG] Channel mapping - original: {provider} -> normalized: {normalized_channel}")

    # ===== 默认平台（SiliconFlow/Kolors）自动注入（前端不可见）=====
    # 触发条件：provider 明确为“默认”或“default”等
    def _is_default_provider(p: Optional[str]) -> bool:
        if not isinstance(p, str):
            return False
        pl = p.strip().lower()
        return pl in ("默认", "default", "default_image", "siliconflow", "siliconflow_default")
    if _is_default_provider(provider):
        # 地址与模型固定，密钥从本地环境注入；严禁将密钥写入仓库
        if SILICONFLOW_IMAGE_API_URL:
            effective_url = SILICONFLOW_IMAGE_API_URL
        if SILICONFLOW_DEFAULT_IMAGE_MODEL:
            effective_model = SILICONFLOW_DEFAULT_IMAGE_MODEL
            model_lower = (effective_model or "").lower()
        if SILICONFLOW_API_KEY_DEFAULT:
            effective_api_key = SILICONFLOW_API_KEY_DEFAULT
        url = effective_url
        headers = {
            "Authorization": f"Bearer {effective_api_key or ''}",
            "Content-Type": "application/json",
            "User-Agent": "EzTalkProxy/1.9.9",
            "Accept": "application/json"
        }
        logger.info(f"[IMG DEFAULT] Using SiliconFlow defaults. url={effective_url}, model={effective_model}")
    
    # ==== Doubao Seedream（即梦4.0）自动适配（OpenAI兼容分支中拦截重写）====
    try:
        api_addr_lower = (url or "").lower()
        is_seedream_model = (
            ("doubao" in model_lower) or
            ("seedream" in model_lower) or
            ("volces.com" in api_addr_lower)
        )
    except Exception:
        is_seedream_model = False

    if is_seedream_model and normalized_channel != "gemini":
        # 规范上游地址（若用户误带了 '#/v1/images/generations' 等）
        try:
            if isinstance(url, str) and "#" in url:
                url = url.split("#", 1)[0]
        except Exception:
            pass
        # 若未提供完整地址，则使用火山方舟官方端点
        if not isinstance(url, str) or not url.strip():
            url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

        # 构造 Seedream 上游 payload，仅包含被支持字段，移除 guidance_scale/steps/batch 等
        seedream_payload: Dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt or "",
            # 默认按 URL 返回，兼容上游
            "response_format": "url",
        }

        # 透传 response_format / stream / watermark（如有）
        try:
            if isinstance(getattr(request, "response_format", None), str) and request.response_format:
                seedream_payload["response_format"] = request.response_format
        except Exception:
            pass
        try:
            if isinstance(getattr(request, "stream", None), bool):
                seedream_payload["stream"] = request.stream
        except Exception:
            pass
        try:
            if isinstance(getattr(request, "watermark", None), bool):
                seedream_payload["watermark"] = request.watermark
        except Exception:
            pass

        # 透传组图/连续生成控制参数（如有）
        try:
            if isinstance(getattr(request, "sequential_image_generation", None), str) and request.sequential_image_generation:
                seedream_payload["sequential_image_generation"] = request.sequential_image_generation
        except Exception:
            pass
        try:
            if isinstance(getattr(request, "sequential_image_generation_options", None), dict) and request.sequential_image_generation_options:
                seedream_payload["sequential_image_generation_options"] = request.sequential_image_generation_options
        except Exception:
            pass

        # 优先使用顶层 size；否则将通用 image_size 映射为 size
        explicit_size = False
        try:
            if isinstance(getattr(request, "size", None), str) and request.size.strip():
                seedream_payload["size"] = request.size.strip()
                explicit_size = True
            elif isinstance(request.image_size, str) and request.image_size.strip():
                seedream_payload["size"] = request.image_size.strip()
        except Exception:
            pass

        # 若未显式传 size，且 size 缺失或仍为默认 1024x1024，则默认提升到 2K（与官方文档一致，避免低清默认）
        try:
            sz_val = seedream_payload.get("size")
            if not explicit_size and (not isinstance(sz_val, str) or not sz_val.strip() or sz_val.strip().lower() == "1024x1024"):
                seedream_payload["size"] = "2K"
                logger.info("[IMG SEEDREAM] Size not explicitly provided; defaulting to 2K for better quality.")
        except Exception:
            pass

        # 若提供了宽高比（aspectRatio），将 2K/4K 别名或默认 2K 细化为与比例一致的像素 WxH
        try:
            ratio_val = None
            try:
                ratio_val = getattr(request, "aspect_ratio", None)
            except Exception:
                ratio_val = None
            if not isinstance(ratio_val, str) or not ratio_val.strip():
                gc = getattr(request, "generation_config", None)
                if isinstance(gc, dict):
                    img_cfg = gc.get("imageConfig") or {}
                    ar = img_cfg.get("aspectRatio")
                    if isinstance(ar, str) and ar.strip():
                        ratio_val = ar.strip()

            def _map_seedream_size(alias_or_wh: str, ratio: str) -> str:
                if not isinstance(alias_or_wh, str) or not alias_or_wh.strip():
                    return alias_or_wh
                s = alias_or_wh.strip().lower()
                # 如果已经是 WxH 形式，直接返回
                if "x" in s and all(part.isdigit() for part in s.split("x", 1)):
                    return alias_or_wh.strip()

                # 标准化比例
                r = (ratio or "").strip()
                # 合法的 Seedream 预设表
                preset_2k = {
                    "1:1": "2048x2048",
                    "16:9": "2048x1152",
                    "9:16": "1152x2048",
                    "4:3": "2048x1536",
                    "3:4": "1536x2048",
                }
                preset_4k = {
                    "1:1": "4096x4096",
                    "16:9": "3840x2160",
                    "9:16": "2160x3840",
                    "4:3": "4096x3072",
                    "3:4": "3072x4096",
                }
                if s == "2k":
                    return preset_2k.get(r, "2048x2048")
                if s == "4k":
                    return preset_4k.get(r, "4096x4096")
                # 未知别名：保持原值
                return alias_or_wh.strip()

            if isinstance(ratio_val, str) and ratio_val.strip():
                current_size = seedream_payload.get("size")
                # 对 2K/4K 或默认的 2K 进行细化
                mapped = _map_seedream_size(current_size or "2K", ratio_val)
                if mapped and mapped != current_size:
                    seedream_payload["size"] = mapped
                    logger.info(f"[IMG SEEDREAM] Refined size by aspectRatio='{ratio_val}': {current_size} -> {mapped}")
        except Exception as _e_map:
            logger.warning(f"[IMG SEEDREAM] Aspect-ratio size refinement skipped due to error: {_e_map}")

        # 收集参考图片：优先读取顶层 image（数组）；并兼容从 contents 中抽取可能的图片 URL
        img_urls: list[str] = []
        try:
            top_images = getattr(request, "image", None)
            if isinstance(top_images, list):
                for u in top_images:
                    if isinstance(u, str) and u:
                        img_urls.append(u)
        except Exception:
            pass
        try:
            if isinstance(request.contents, list) and request.contents:
                for part in request.contents:
                    if isinstance(part, dict):
                        if isinstance(part.get("image_url"), dict) and isinstance(part["image_url"].get("url"), str):
                            img_urls.append(part["image_url"]["url"])
                        elif isinstance(part.get("url"), str):
                            img_urls.append(part["url"])
        except Exception:
            pass
        # 去重并写入
        try:
            if img_urls:
                dedup = []
                seen = set()
                for u in img_urls:
                    if u not in seen:
                        seen.add(u)
                        dedup.append(u)
                if dedup:
                    seedream_payload["image"] = dedup
        except Exception:
            pass

        payload = seedream_payload
        seedream_mode = True
        # 跳过后续 Gemini/默认分支，进入统一下游请求流程
    # ==== Gemini 渠道强制走原生（与前端/配置一致），不再依赖 host 判断 ====
    if is_gemini_image_model and normalized_channel == "gemini":
        # Use Google's native Gemini API format
        logger.info(f"[IMG] Using Google native API format for {effective_model} (provider: {provider})")
        
        # 同时提供 Authorization 与 x-goog-api-key，最大化兼容第三方聚合
        headers = {
            "Authorization": f"Bearer {effective_api_key or ''}",
            "Content-Type": "application/json",
            "User-Agent": "EzTalkProxy/1.9.9",
            "Accept": "application/json",
            "x-goog-api-key": effective_api_key or ""
        }
        
        # ===== Build current user parts and capture user text for history =====
        content_parts = []
        current_user_text = ""
        if request.contents:  # Image editing mode with input images
            # Add text prompt first
            text_prompt = ""
            for part in request.contents:
                if "text" in part and part["text"]:
                    text_prompt = part["text"]
                    break
            if text_prompt:
                content_parts.append({"text": text_prompt})
                current_user_text = text_prompt
            else:
                default_text = request.prompt or "Generate an image based on the provided image."
                content_parts.append({"text": default_text})
                current_user_text = default_text
            # Add inline images
            for part in request.contents:
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    mime_type = inline_data.get("mime_type", "image/jpeg")
                    b64_data = inline_data.get("data", "")
                    if b64_data:
                        content_parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data
                            }
                        })
        else:  # Pure text-to-image mode
            user_text = request.prompt or ""
            content_parts = [{"text": user_text}]
            current_user_text = user_text

        # ===== Inject session history turns (text-only) before current turn =====
        history_contents = _get_history_for_contents(session_key) if enable_session and session_key else []
        if history_contents:
            logger.info(f"[IMG] Injecting history turns: {len(history_contents)} for session {session_key[:8]}...")

        # If there is a previous image in session and current request didn't provide any image,
        # attach it as a reference image to enhance continuity. Supports data URI and http(s) URL.
        try:
            has_inline_image_in_request = any(("inline_data" in p) for p in (request.contents or []))
        except Exception:
            has_inline_image_in_request = False

        last_image_ref = _SESSION_LAST_IMAGE.get(session_key) if enable_session and session_key else None
        if last_image_ref and not has_inline_image_in_request:
            try:
                # Case A: data URI already cached
                if isinstance(last_image_ref, str) and last_image_ref.startswith("data:") and ";base64," in last_image_ref:
                    mime_type = last_image_ref[5:last_image_ref.index(";base64,")]
                    b64_data = last_image_ref.split(";base64,", 1)[1]
                    content_parts.append({
                        "inline_data": {
                            "mime_type": mime_type or "image/png",
                            "data": b64_data
                        }
                    })
                    logger.info(f"[IMG] Attached previous data-URI image as reference. mime={mime_type}, b64_len={len(b64_data)}")
                # Case B: URL cached -> fetch and convert to base64 for inline_data
                elif isinstance(last_image_ref, str) and last_image_ref.startswith(("http://", "https://")):
                    try:
                        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0), http2=True, follow_redirects=True) as _client:
                            resp_img = await _client.get(last_image_ref)
                            resp_img.raise_for_status()
                            img_bytes = resp_img.content
                            mime = resp_img.headers.get("Content-Type", "image/png")
                        import base64 as _b64
                        b64_data = _b64.b64encode(img_bytes).decode("utf-8")
                        content_parts.append({
                            "inline_data": {
                                "mime_type": mime or "image/png",
                                "data": b64_data
                            }
                        })
                        # 将下载后的 data URI 重新写回缓存，后续无需再拉取
                        _SESSION_LAST_IMAGE[session_key] = f"data:{mime};base64,{b64_data}"
                        logger.info(f"[IMG] Fetched previous URL image and attached as reference. mime={mime}, bytes={len(img_bytes)}")
                    except Exception as fetch_err:
                        logger.warning(f"[IMG] Failed to fetch last image URL for reference: {fetch_err}")
            except Exception as ref_err:
                logger.warning(f"[IMG] Failed to attach previous image reference: {ref_err}")

        contents_list = []
        contents_list.extend(history_contents)
        contents_list.append({"role": "user", "parts": content_parts})

        # Debug: summarize contents (turns/parts/inline_data count)
        try:
            total_turns = len(contents_list)
            inline_cnt = 0
            text_cnt = 0
            for c in contents_list:
                for p in c.get("parts", []):
                    if "inline_data" in p:
                        inline_cnt += 1
                    if "text" in p:
                        text_cnt += 1
            logger.info(f"[IMG DEBUG] Gemini payload summary: turns={total_turns}, text_parts={text_cnt}, inline_data_parts={inline_cnt}")
        except Exception:
            pass

        # Construct Google native API payload
        payload = {
            "contents": contents_list
        }
        
        # Add generation config (align with Google Gemini docs)
        generation_config: Dict[str, Any] = {}

        # 从顶层或 generationConfig 里兜底读取 responseModalities / aspectRatio
        resp_modalities: Optional[List[str]] = None
        aspect_ratio_val: Optional[str] = None
        try:
            resp_modalities = getattr(request, "response_modalities", None)
        except Exception:
            resp_modalities = None
        try:
            aspect_ratio_val = getattr(request, "aspect_ratio", None)
        except Exception:
            aspect_ratio_val = None

        # 若顶层无值，从原样透传的 generationConfig 中读
        try:
            if not resp_modalities and isinstance(request.generation_config, dict):
                maybe = request.generation_config.get("responseModalities")
                if isinstance(maybe, list) and maybe:
                    resp_modalities = maybe
        except Exception:
            pass
        try:
            if not aspect_ratio_val and isinstance(request.generation_config, dict):
                img_cfg = request.generation_config.get("imageConfig")
                if isinstance(img_cfg, dict):
                    ar = img_cfg.get("aspectRatio")
                    if isinstance(ar, str) and ar:
                        aspect_ratio_val = ar
        except Exception:
            pass

        # 写入 generationConfig
        if resp_modalities:
            generation_config["responseModalities"] = list(resp_modalities)
        if aspect_ratio_val:
            generation_config.setdefault("imageConfig", {})
            generation_config["imageConfig"]["aspectRatio"] = aspect_ratio_val  # e.g. "16:9"

        # 兼容性处理：对非 Google 官方端点移除 responseModalities，避免第三方代理返回 400
        try:
            if not _is_google_official_api(request.apiAddress) and "responseModalities" in generation_config:
                generation_config.pop("responseModalities", None)
                logger.info("[IMG] Removed responseModalities for non-official Gemini endpoint to improve compatibility.")
        except Exception:
            pass

        # 注意：generateContent 不需要 candidate_count/image_size 之类占位，避免 400
        if generation_config:
            payload["generationConfig"] = generation_config
            
        # Construct the correct Google API URL
        model_name = effective_model
        # Remove the /v1/images/generations suffix if present and replace with Google's format
        base_url = url
        if base_url.endswith('/v1/images/generations'):
            base_url = base_url[:-len('/v1/images/generations')]
        elif base_url.endswith('/v1/images/generations/'):
            base_url = base_url[:-len('/v1/images/generations/')]
        
        if not base_url.endswith('/'):
            base_url += '/'
        url = f"{base_url}v1beta/models/{model_name}:generateContent"
        # 追加 ?key= 以兼容官方与部分聚合实现
        delimiter = '&' if '?' in url else '?'
        if "key=" not in url:
            url = f"{url}{delimiter}key={effective_api_key or ''}"
        # 调试打印最终生成配置，便于核对是否成功带上宽高比
        try:
            logger.info(f"[IMG] Gemini native payload.generationConfig = {payload.get('generationConfig')}")
        except Exception:
            pass

        # 预构建一个“OpenAI兼容”降级payload（用于非官方端点且无图时的兜底重试）
        compat_payload_for_gemini = None
        try:
            content_parts = []
            if request.contents:
                # 编辑模式，提取文本与图像
                text_prompt = ""
                for part in request.contents:
                    if "text" in part and part["text"]:
                        text_prompt = part["text"]
                        break
                if text_prompt:
                    content_parts.append({"type": "text", "text": text_prompt})
                else:
                    content_parts.append({"type": "text", "text": "Edit the image."})
                for part in request.contents:
                    if "inline_data" in part:
                        inline_data = part["inline_data"]
                        mime_type = inline_data.get("mime_type", "image/jpeg")
                        b64_data = inline_data.get("data", "")
                        if b64_data:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                            })
            else:
                content_parts = [{"type": "text", "text": request.prompt or ""}]
            compat_payload_for_gemini = {
                "model": effective_model,
                "messages": [{"role": "user", "content": content_parts}],
                "stream": False,
                "modalities": ["image"]
            }
            # 透传宽高比
            try:
                if getattr(request, "aspect_ratio", None):
                    compat_payload_for_gemini["aspect_ratio"] = request.aspect_ratio
                elif isinstance(request.generation_config, dict):
                    img_cfg = request.generation_config.get("imageConfig") or {}
                    ar = img_cfg.get("aspectRatio")
                    if isinstance(ar, str) and ar:
                        compat_payload_for_gemini["aspect_ratio"] = ar
            except Exception:
                pass
        except Exception:
            compat_payload_for_gemini = None
    elif is_gemini_image_model:
        # For OpenAI compatible format (even for Gemini models when provider is not "gemini")
        logger.info(f"[IMG] Using OpenAI-compatible format for Gemini model {effective_model} (provider: {provider})")
        if "/images/generations" in url:
            url = url.replace("/images/generations", "/chat/completions")

        content_parts = []
        # 图像生成或编辑的核心逻辑
        if request.contents:  # 这是图像编辑模式
            text_prompt = ""
            # 首先找到文本部分
            for part in request.contents:
                if "text" in part and part["text"]:
                    text_prompt = part["text"]
                    break
            
            # OpenRouter文档要求文本部分在前
            if text_prompt:
                content_parts.append({"type": "text", "text": text_prompt})
            else:  # 如果没有文本，提供一个默认的
                content_parts.append({"type": "text", "text": "Edit the image."})
            
            # 然后添加图像部分
            for part in request.contents:
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    mime_type = inline_data.get("mime_type", "image/jpeg")
                    b64_data = inline_data.get("data", "")
                    if b64_data:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                        })
            
            payload = {
                "model": effective_model,
                "messages": [{"role": "user", "content": content_parts}],
                "stream": False,
                "modalities": ["image"]
            }
            # 尝试透传宽高比（部分聚合商支持顶层 aspect_ratio）
            try:
                if getattr(request, "aspect_ratio", None):
                    payload["aspect_ratio"] = request.aspect_ratio
                elif isinstance(request.generation_config, dict):
                    img_cfg = request.generation_config.get("imageConfig") or {}
                    ar = img_cfg.get("aspectRatio")
                    if isinstance(ar, str) and ar:
                        payload["aspect_ratio"] = ar
            except Exception:
                pass
        else:  # 这是纯文本图像生成模式
            # 使用多模态 parts 格式并显式声明 modalities，以触发中转商的图片输出
            content_parts = [{"type": "text", "text": request.prompt or ""}]
            payload = {
                "model": effective_model,
                "messages": [{"role": "user", "content": content_parts}],
                "stream": False,
                "modalities": ["image"]
            }
            # 同样在 OpenAI 兼容分支尽可能附带宽高比，提升兼容性
            try:
                if getattr(request, "aspect_ratio", None):
                    payload["aspect_ratio"] = request.aspect_ratio
                elif isinstance(request.generation_config, dict):
                    img_cfg = request.generation_config.get("imageConfig") or {}
                    ar = img_cfg.get("aspectRatio")
                    if isinstance(ar, str) and ar:
                        payload["aspect_ratio"] = ar
            except Exception:
                pass
    elif not seedream_mode:
        payload = request.model_dump(exclude={"apiAddress", "apiKey", "contents"})
        # 强制以有效模型为准（支持“默认平台”覆盖）
        try:
            payload["model"] = effective_model
        except Exception:
            pass
        try:
            img_size = payload.get("image_size")
            if not isinstance(img_size, str) or not img_size.strip() or "<" in img_size:
                payload["image_size"] = "1024x1024"
        except Exception:
            payload["image_size"] = "1024x1024"

    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        size_for_log = payload.get("size") if seedream_mode else payload.get("image_size")
        logger.info(f"[IMG] Proxying to upstream: {url} | model={payload.get('model')} | size={size_for_log} | batch={payload.get('batch_size')} | steps={payload.get('num_inference_steps')} | guidance={payload.get('guidance_scale')}")
    except Exception:
        logger.info(f"[IMG] Proxying to upstream: {url} | model={payload.get('model')}")
    logger.debug(f"[IMG] Upstream payload: {payload}")
    if "x-goog-api-key" in headers and "Authorization" in headers:
        logger.info(f"[IMG] Request headers: Authorization & x-goog-api-key present, Content-Type: application/json")
    elif "x-goog-api-key" in headers:
        logger.info(f"[IMG] Request headers: x-goog-api-key: {(effective_api_key[:10] + '...') if isinstance(effective_api_key, str) and effective_api_key else '(none)'} , Content-Type: application/json")
    else:
        logger.info(f"[IMG] Request headers: Authorization: Bearer {(effective_api_key[:10] + '...') if isinstance(effective_api_key, str) and effective_api_key else '(none)'} , Content-Type: application/json")
 
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
            resp = await client.post(url, headers=headers, json=payload)
    except httpx.RequestError as e:
        logger.error(f"[IMG] Upstream request error to {url}: {e}", exc_info=True)
        return _fallback_response(f"request_error: {e}", user_text="网络异常：无法连接上游服务，请稍后重试。")

    # 非 2xx：将上游响应体传回，便于前端/日志定位
    if resp.status_code < 200 or resp.status_code >= 300:
        text_preview = resp.text[:1000] if resp.text else "(empty)"
        logger.error(f"[IMG] Upstream non-2xx {resp.status_code}. Body preview: {text_preview}")
        logger.error(f"[IMG] Response headers: {dict(resp.headers)}")
        
        # 若为 Gemini 原生分支，做针对性回退处理（400/401）
        try:
            model_lower = request.model.lower()
            provider_lower = (request.provider or "").lower()
            is_gemini_image_model = "gemini" in model_lower and ("flash-image" in model_lower or "gemini-pro-vision" in model_lower)
            normalized_channel = {"gemini": "gemini", "google": "gemini", "openai compatible": "openai_compatible", "openai": "openai_compatible", "OpenAI兼容": "openai_compatible"}.get(request.provider or "openai compatible", provider_lower)
        except Exception:
            normalized_channel = provider_lower
            is_gemini_image_model = False

        # 400 invalid argument 时，尝试移除 responseModalities 仅保留 imageConfig.aspectRatio 再试一次
        if resp.status_code == 400 and is_gemini_image_model and normalized_channel == "gemini":
            # First attempt: retry once without history (possible context-length issue)
            try:
                if isinstance(payload.get("contents"), list) and payload["contents"]:
                    # Build a no-history payload keeping only current user turn
                    last_turn = payload["contents"][-1]
                    payload_no_history = dict(payload)
                    payload_no_history["contents"] = [last_turn]
                    logger.info("[IMG] 400 from upstream in Gemini native branch, retrying once without history...")
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
                        alt_no_hist = await client.post(url, headers=headers, json=payload_no_history)
                    if 200 <= alt_no_hist.status_code < 300:
                        try:
                            raw_alt_no_hist = alt_no_hist.json()
                            logger.info(f"[IMG] Retry without history succeeded.")
                            normalized_no_hist = _normalize_response(raw_alt_no_hist, append_failure_hint=False)
                            # 回写本轮会话历史（Gemini 原生）
                            try:
                                if enable_session and session_key and is_gemini_image_model and normalized_channel == "gemini":
                                    model_text = normalized_no_hist.text or "Image generated."
                                    if 'current_user_text' in locals() and current_user_text is not None:
                                        _append_turn(session_key, "user", current_user_text)
                                    _append_turn(session_key, "model", model_text)
                                    # 存储参考图
                                    try:
                                        first_url = normalized_no_hist.images[0].url if normalized_no_hist.images else None
                                        if isinstance(first_url, str):
                                            _SESSION_LAST_IMAGE[session_key] = first_url
                                            if first_url.startswith("data:"):
                                                logger.info("[IMG] Stored last generated image (retry-no-history, data URI).")
                                            else:
                                                logger.info("[IMG] Stored last generated image (retry-no-history, URL).")
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            if getattr(request, "force_data_uri", False):
                                normalized_no_hist = await _force_images_to_data_uri(normalized_no_hist)
                            try:
                                conv_for_save = persist_key
                                if conv_for_save and getattr(normalized_no_hist, "images", None):
                                    images_payload = [{"url": img.url} for img in normalized_no_hist.images if getattr(img, "url", None)]
                                    meta_payload = {"text": normalized_no_hist.text, "seed": normalized_no_hist.seed, "timings": getattr(normalized_no_hist, "timings", None)}
                                    logger.info(f"[IMG HIST] Saving {len(images_payload)} image(s) for conversation={conv_for_save} (no-history retry)")
                                    save_images(conv_for_save, images_payload, meta_payload)
                            except Exception as _hist_err:
                                logger.warning(f"[IMG HIST] Save failed (no-history retry): {_hist_err}")
                            return normalized_no_hist
                        except Exception as e_norm1:
                            logger.error(f"[IMG] Retry (no history) succeeded but normalization failed: {e_norm1}", exc_info=True)
                            return _fallback_response(f"normalize_error_after_400_retry_no_history: {e_norm1}", user_text="图像生成请求成功，但响应数据解析失败。请稍后重试或联系技术支持。")
                    else:
                        logger.warning(f"[IMG] Retry without history still non-2xx: {alt_no_hist.status_code}. Body: {alt_no_hist.text[:500] if alt_no_hist.text else '(empty)'}")
            except Exception as e_nohist:
                logger.error(f"[IMG] Error during 400 retry without history: {e_nohist}", exc_info=True)

            # Second attempt: existing fallback removing responseModalities
            try:
                gc = payload.get("generationConfig") if isinstance(payload, dict) else None
                if isinstance(gc, dict) and "responseModalities" in gc:
                    gc_retry = dict(gc)
                    gc_retry.pop("responseModalities", None)
                    payload_retry = dict(payload)
                    payload_retry["generationConfig"] = gc_retry
                    logger.info("[IMG] 400 from upstream in Gemini native branch, retrying without responseModalities...")
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
                        alt = await client.post(url, headers=headers, json=payload_retry)
                    if 200 <= alt.status_code < 300:
                        try:
                            raw_alt = alt.json()
                            logger.info(f"[IMG] Retry without responseModalities succeeded. Returning normalized result.")
                            normalized_alt = _normalize_response(raw_alt, append_failure_hint=False)
                            # 回写本轮会话历史（Gemini 原生）
                            try:
                                if enable_session and session_key and is_gemini_image_model and normalized_channel == "gemini":
                                    model_text = normalized_alt.text or "Image generated."
                                    if 'current_user_text' in locals() and current_user_text is not None:
                                        _append_turn(session_key, "user", current_user_text)
                                    _append_turn(session_key, "model", model_text)
                                    # 存储参考图
                                    try:
                                        first_url = normalized_alt.images[0].url if normalized_alt.images else None
                                        if isinstance(first_url, str):
                                            _SESSION_LAST_IMAGE[session_key] = first_url
                                            if first_url.startswith("data:"):
                                                logger.info("[IMG] Stored last generated image (retry-no-responseModalities, data URI).")
                                            else:
                                                logger.info("[IMG] Stored last generated image (retry-no-responseModalities, URL).")
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            if getattr(request, "force_data_uri", False):
                                normalized_alt = await _force_images_to_data_uri(normalized_alt)
                            try:
                                conv_for_save = persist_key
                                if conv_for_save and getattr(normalized_alt, "images", None):
                                    images_payload = [{"url": img.url} for img in normalized_alt.images if getattr(img, "url", None)]
                                    meta_payload = {"text": normalized_alt.text, "seed": normalized_alt.seed, "timings": getattr(normalized_alt, "timings", None)}
                                    logger.info(f"[IMG HIST] Saving {len(images_payload)} image(s) for conversation={conv_for_save} (alt retry)")
                                    save_images(conv_for_save, images_payload, meta_payload)
                            except Exception as _hist_err:
                                logger.warning(f"[IMG HIST] Save failed (alt retry): {_hist_err}")
                            return normalized_alt
                        except Exception as e_norm:
                            logger.error(f"[IMG] Retry succeeded but normalization failed: {e_norm}", exc_info=True)
                            return _fallback_response(f"normalize_error_after_400_retry: {e_norm}", user_text="图像生成请求成功，但响应数据格式异常。请检查API配置或稍后重试。")
                    else:
                        logger.warning(f"[IMG] Retry without responseModalities still non-2xx: {alt.status_code}. Body: {alt.text[:500] if alt.text else '(empty)'}")
            except Exception as e_retry:
                logger.error(f"[IMG] Error during 400 fallback retry: {e_retry}", exc_info=True)
        
        if resp.status_code == 401 and is_gemini_image_model and normalized_channel == "gemini":
            logger.warning("[IMG] 401 from upstream in Gemini native branch, trying alternative auth strategies for compatibility...")
            # 生成三个尝试：
            # A) 仅 x-goog-api-key + URL ?key=    B) 仅 Authorization（移除 ?key）    C) x-api-key 头
            alt_attempts = []
            
            # 准备 URL 变体
            def remove_key_param(u: str) -> str:
                try:
                    # 移除 ?key= 或 &key=
                    return re.sub(r'([?&])key=[^&]*(&)?', lambda m: '?' if m.group(2) else '', u).rstrip('?')
                except Exception:
                    return u
            
            url_with_key = url if re.search(r'([?&])key=', url) else (f"{url}{'&' if '?' in url else '?'}key={effective_api_key or ''}")
            url_without_key = remove_key_param(url)
            
            # A) 仅 x-goog-api-key
            a_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            a_headers["x-goog-api-key"] = effective_api_key or ""
            alt_attempts.append(("A:x-goog-only", url_with_key, a_headers))
            
            # B) 仅 Authorization
            b_headers = {k: v for k, v in headers.items() if k.lower() != "x-goog-api-key"}
            b_headers["Authorization"] = f"Bearer {effective_api_key or ''}"
            alt_attempts.append(("B:auth-only", url_without_key, b_headers))
            
            # C) x-api-key
            c_headers = {k: v for k, v in headers.items() if k.lower() not in ("x-goog-api-key",)}
            c_headers["x-api-key"] = effective_api_key or ""
            alt_attempts.append(("C:x-api-key", url_without_key, c_headers))
            
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
                    for idx, (label, alt_url, alt_headers) in enumerate(alt_attempts):
                        logger.info(f"[IMG] Retrying ({label}) -> {alt_url}")
                        alt_resp = await client.post(alt_url, headers=alt_headers, json=payload)
                        if 200 <= alt_resp.status_code < 300:
                            try:
                                raw = alt_resp.json()
                                raw_for_log = str(raw)
                                log_preview = raw_for_log[:1000] + ('...' if len(raw_for_log) > 1000 else '')
                                logger.info(f"[IMG] Alternative attempt succeeded ({label}). RAW preview: {log_preview}")
                                normalized = _normalize_response(raw, append_failure_hint=False)
                                if getattr(normalized, 'images', []) and len(normalized.images) > 0:
                                    logger.info(f"[IMG] Image generation normalized successfully after alternative attempt. Text: {normalized.text}, Images: {len(normalized.images)}")
                                    # 回写本轮会话历史（Gemini 原生）
                                    try:
                                        if enable_session and session_key and is_gemini_image_model and normalized_channel == "gemini":
                                            model_text = normalized.text or "Image generated."
                                            if 'current_user_text' in locals() and current_user_text is not None:
                                                _append_turn(session_key, "user", current_user_text)
                                            _append_turn(session_key, "model", model_text)
                                            # 存储参考图
                                            try:
                                                first_url = normalized.images[0].url if normalized.images else None
                                                if isinstance(first_url, str):
                                                    _SESSION_LAST_IMAGE[session_key] = first_url
                                                    if first_url.startswith("data:"):
                                                        logger.info("[IMG] Stored last generated image (alt-auth, data URI).")
                                                    else:
                                                        logger.info("[IMG] Stored last generated image (alt-auth, URL).")
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                                    if getattr(request, "force_data_uri", False):
                                        normalized = await _force_images_to_data_uri(normalized)
                                    try:
                                        conv_for_save = persist_key
                                        if conv_for_save and getattr(normalized, "images", None):
                                            images_payload = [{"url": img.url} for img in normalized.images if getattr(img, "url", None)]
                                            meta_payload = {"text": normalized.text, "seed": normalized.seed, "timings": getattr(normalized, "timings", None)}
                                            logger.info(f"[IMG HIST] Saving {len(images_payload)} image(s) for conversation={conv_for_save} (alt auth)")
                                            save_images(conv_for_save, images_payload, meta_payload)
                                    except Exception as _hist_err:
                                        logger.warning(f"[IMG HIST] Save failed (alt auth): {_hist_err}")
                                    return normalized
                                # 若本次规范化仍无图片，只有在最后一次替代尝试后才附加失败提示
                                if idx == len(alt_attempts) - 1:
                                    # 所有替代鉴权重试结束仍无图片：返回统一错误文本（不返回纯文本描述）
                                    error_msg = "图片生成失败或被上游拒绝，请检查API Key/供应商兼容性或稍后重试。"
                                    logger.info(f"[IMG] Alternative attempts exhausted without images. Returning error text.")
                                    return ImageGenerationResponse(images=[], text=error_msg, timings={"inference": 0}, seed=random.randint(1, 2**31 - 1))
                                # 否则继续尝试下一种认证策略
                            except Exception as e_norm:
                                logger.error(f"[IMG] Alternative attempt succeeded but normalization failed: {e_norm}", exc_info=True)
                                return _fallback_response(f"normalize_error_after_alt: {e_norm}", user_text="图像生成成功但响应解析失败。请检查API Key和供应商配置是否正确。")
                        else:
                            logger.warning(f"[IMG] Alternative attempt ({label}) still non-2xx: {alt_resp.status_code}. Body: {alt_resp.text[:500] if alt_resp.text else '(empty)'}")
            except Exception as e_alt:
                logger.error(f"[IMG] Error during alternative auth retries: {e_alt}", exc_info=True)
        
        # 将上游错误提示透传给前端（例如地区限制）
        user_text = None
        try:
            err_json = resp.json()
            err_obj = err_json.get("error", {}) if isinstance(err_json, dict) else {}
            raw_msg = err_obj.get("message")
            raw_embedded = None
            meta = err_obj.get("metadata") if isinstance(err_obj, dict) else None
            if isinstance(meta, dict):
                raw_embedded = meta.get("raw")
            candidate_texts = [raw_msg, raw_embedded, text_preview]
            for ct in candidate_texts:
                if isinstance(ct, str) and ct:
                    if "User location is not supported" in ct or "FAILED_PRECONDITION" in ct:
                        user_text = "区域限制：上游拒绝该模型的调用。请更换可用地区节点/供应商，或选择其他模型。"
                        break
            if user_text is None and isinstance(raw_msg, str) and raw_msg:
                user_text = f"上游错误（{resp.status_code}）：{raw_msg}"
        except Exception:
            if "User location is not supported" in text_preview:
                user_text = "区域限制：上游拒绝该模型的调用。请更换可用地区节点/供应商，或选择其他模型。"
            else:
                user_text = f"上游错误（{resp.status_code}）：{text_preview}"
        # 返回兜底结构（HTTP 200 由路由层自动处理，因为我们返回模型对象）
        return _fallback_response(f"upstream_{resp.status_code}: {text_preview}", user_text=user_text)

    try:
        raw = resp.json()
        # 日志记录截断的响应，避免日志过长
        raw_for_log = str(raw)
        log_preview = raw_for_log[:1000] + ('...' if len(raw_for_log) > 1000 else '')
        logger.info(f"[IMG] Upstream RAW response from provider (preview): {log_preview}")
    except Exception as e:
        logger.error(f"[IMG] Upstream returned non-JSON body: {e}. Body preview: {resp.text[:500]}", exc_info=True)
        return _fallback_response(f"non_json_upstream: {e}", user_text=f"上游返回非JSON响应：{str(e)}")

    try:
        normalized = _normalize_response(raw, append_failure_hint=False)
        if getattr(normalized, 'images', []) and len(normalized.images) > 0:
            logger.info(f"[IMG] Image generation normalized successfully. Text: {normalized.text}, Images: {len(normalized.images)}")
            # ===== Append current turn to session history (Gemini native only) =====
            try:
                if enable_session and session_key and is_gemini_image_model and normalized_channel == "gemini":
                    model_text = normalized.text or "Image generated."
                    if current_user_text is not None:
                        _append_turn(session_key, "user", current_user_text)
                    _append_turn(session_key, "model", model_text)
                    # store last image data URI if available
                    try:
                        first_url = normalized.images[0].url if normalized.images else None
                        if isinstance(first_url, str):
                            # 支持 data URI 与 http(s) URL 两种缓存形式
                            _SESSION_LAST_IMAGE[session_key] = first_url
                            if first_url.startswith("data:"):
                                logger.info("[IMG] Stored last generated image as data URI for continuity.")
                            else:
                                logger.info("[IMG] Stored last generated image URL for continuity.")
                    except Exception:
                        pass
            except Exception:
                pass
            if getattr(request, "force_data_uri", False):
                normalized = await _force_images_to_data_uri(normalized)
            try:
                conv_for_save = persist_key
                if conv_for_save and getattr(normalized, "images", None):
                    images_payload = [{"url": img.url} for img in normalized.images if getattr(img, "url", None)]
                    meta_payload = {"text": normalized.text, "seed": normalized.seed, "timings": getattr(normalized, "timings", None)}
                    logger.info(f"[IMG HIST] Saving {len(images_payload)} image(s) for conversation={conv_for_save}")
                    save_images(conv_for_save, images_payload, meta_payload)
            except Exception as _hist_err:
                logger.warning(f"[IMG HIST] Save failed: {_hist_err}")
            return normalized
        # 轻量级重试：首次规范化无图片、且非内容拦截时，重试一次
        if (normalized.text and not str(normalized.text).startswith("[CONTENT_FILTER]")):
            # 连续重试（不回传纯文本，只有生成出图片才返回）
            try:
                max_retries = 2  # 在首次失败后再重试2次，共3次尝试
                last_json = raw
                for attempt in range(1, max_retries + 1):
                    logger.info(f"[IMG] No images found; performing retry {attempt}/{max_retries} without user-visible text...")
                    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client:
                        resp2 = await client.post(url, headers=headers, json=payload)
                    if 200 <= resp2.status_code < 300:
                        try:
                            raw2 = resp2.json()
                            last_json = raw2
                            raw2_for_log = str(raw2)
                            log2_preview = raw2_for_log[:1000] + ('...' if len(raw2_for_log) > 1000 else '')
                            logger.info(f"[IMG] Upstream RAW response from provider (retry preview): {log2_preview}")
                            normalized2 = _normalize_response(raw2, append_failure_hint=False)
                            logger.info(f"[IMG] After retry {attempt}, Text: {normalized2.text}, Images: {len(normalized2.images)}")
                            if getattr(normalized2, 'images', []) and len(normalized2.images) > 0:
                                if getattr(request, "force_data_uri", False):
                                    normalized2 = await _force_images_to_data_uri(normalized2)
                                try:
                                    conv_for_save = persist_key
                                    if conv_for_save and getattr(normalized2, "images", None):
                                        images_payload = [{"url": img.url} for img in normalized2.images if getattr(img, "url", None)]
                                        meta_payload = {"text": normalized2.text, "seed": normalized2.seed, "timings": getattr(normalized2, "timings", None)}
                                        logger.info(f"[IMG HIST] Saving {len(images_payload)} image(s) for conversation={conv_for_save} (retry)")
                                        save_images(conv_for_save, images_payload, meta_payload)
                                except Exception as _hist_err:
                                    logger.warning(f"[IMG HIST] Save failed (retry): {_hist_err}")
                                return normalized2  # 仅当真的生成出图片才返回结果
                            # 若仍无图片，继续下一次重试，不向前端输出纯文本
                        except Exception as e2:
                            logger.error(f"[IMG] Retry {attempt} returned JSON but normalization failed: {e2}", exc_info=True)
                            # 继续下一次重试
                            continue
                    else:
                        logger.warning(f"[IMG] Retry {attempt} non-2xx {resp2.status_code}. Body preview: {resp2.text[:500] if resp2.text else '(empty)'}")
                        # 继续下一次重试
                        continue
                # 所有重试均未拿到图片：仅此时返回错误文本
                error_msg = "图片生成失败或被上游拒绝，请稍后重试或更换模型/供应商。"
                logger.info(f"[IMG] All retries exhausted without images. Returning error text.")
                try:
                    # 若最后一次JSON可解析，尝试附加失败提示（不强制）
                    _ = _normalize_response(last_json, append_failure_hint=True)
                except Exception:
                    pass
                return ImageGenerationResponse(images=[], text=error_msg, timings={"inference": 0}, seed=random.randint(1, 2**31 - 1))
            except Exception as er2:
                logger.error(f"[IMG] Multi-step retry error: {er2}", exc_info=True)
                # 重试过程自身异常，也返回统一错误文本
                error_msg = "图片生成失败或网络异常，请稍后重试或更换模型/供应商。"
                return ImageGenerationResponse(images=[], text=error_msg, timings={"inference": 0}, seed=random.randint(1, 2**31 - 1))
        # 若没有文本可提示或为内容拦截，直接返回追加提示后的结果（追加逻辑内部会自动判断是否需要提示）
        normalized_final = _normalize_response(raw, append_failure_hint=True)
        logger.info(f"[IMG] Returning final normalized result after no-image condition. Text: {normalized_final.text}, Images: {len(normalized_final.images)}")
        return normalized_final
    except Exception as e:
        logger.error(f"[IMG] Failed to normalize upstream response: {e}. Raw keys: {list(raw) if isinstance(raw, dict) else type(raw)}", exc_info=True)
        # 返回兜底结构，避免前端解析失败
        return _fallback_response(f"normalize_error: {e}", user_text="图像生成响应无法识别。上游API返回了不支持的数据格式，请更换模型或供应商后重试。")

# Support both with and without '/chat' prefix to be backward compatible
@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image_generation_v1(payload: Dict[str, Any] = Body(...), request: Request = None, response: Response = None):
    try:
        req = ImageGenerationRequest(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"message": "Invalid image generation request", "errors": e.errors()})
    return await _proxy_and_normalize(req, request_obj=request, response_obj=response)

@router.post("/chat/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image_generation_chat_v1(payload: Dict[str, Any] = Body(...), request: Request = None, response: Response = None):
    try:
        req = ImageGenerationRequest(**payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"message": "Invalid image generation request", "errors": e.errors()})
    return await _proxy_and_normalize(req, request_obj=request, response_obj=response)


@router.get("/v1/images/history")
async def get_image_history_q(conversationId: str):
    """
    Return all persisted image records for a conversationId.
    Payload shape:
    {
      "conversationId": "...",
      "records": [
        { "ts": 1234567890, "images": [{"url":"..."}], "meta": {...} },
        ...
      ]
    }
    """
    if not conversationId:
        raise HTTPException(status_code=400, detail="Missing conversationId")
    records = list_images(conversationId)
    return {"conversationId": conversationId, "records": records}


@router.get("/v1/images/history/{conversation_id}")
async def get_image_history_path(conversation_id: str):
    if not conversation_id:
        raise HTTPException(status_code=400, detail="Missing conversationId")
    records = list_images(conversation_id)
    return {"conversationId": conversation_id, "records": records}