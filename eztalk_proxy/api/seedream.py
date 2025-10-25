import logging
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("EzTalkProxy.Routers.Seedream")
router = APIRouter()


class DoubaoSeedreamRequest(BaseModel):
    # Upstream payload fields (per curl spec)
    model: str = Field(..., description="e.g. doubao-seedream-4-0-250828")
    prompt: str = Field(..., description="text prompt")
    image: Optional[List[str]] = Field(None, description="image URLs for img2img")
    sequential_image_generation: Optional[str] = Field(
        None, description="auto/on/off"
    )
    sequential_image_generation_options: Optional[Dict[str, Any]] = Field(
        None, description="e.g. {\"max_images\": 3}"
    )
    response_format: Optional[str] = Field("url", description="url | b64_json")
    size: Optional[str] = Field(None, description="e.g. 2K | 1024x1024 | 4K")
    stream: Optional[bool] = Field(
        False, description="Passthrough upstream SSE/stream if supported"
    )
    watermark: Optional[bool] = Field(True, description="Enable watermark")

    # Transport fields
    apiAddress: Optional[str] = Field(
        None,
        description="Upstream API base. Defaults to Doubao Seedream endpoint when omitted.",
    )
    apiKey: str = Field(..., description="Bearer token for upstream")

    class Config:
        populate_by_name = True


DEFAULT_SEEDREAM_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"


def _build_upstream_url(api_address: Optional[str]) -> str:
    addr = (api_address or "").strip() if api_address else ""
    if not addr:
        return DEFAULT_SEEDREAM_URL
    # If caller passed host only, normalize path
    # Otherwise respect full path
    return addr


def _make_payload(req: DoubaoSeedreamRequest) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": req.model,
        "prompt": req.prompt,
        "response_format": req.response_format,
        "watermark": req.watermark,
    }
    # Optional passthroughs
    if isinstance(req.image, list) and req.image:
        payload["image"] = req.image
    if req.sequential_image_generation:
        payload["sequential_image_generation"] = req.sequential_image_generation
    if isinstance(req.sequential_image_generation_options, dict):
        payload["sequential_image_generation_options"] = req.sequential_image_generation_options
    if isinstance(req.size, str) and req.size:
        payload["size"] = req.size
    # Note: upstream supports stream flag; we pass it along
    if isinstance(req.stream, bool):
        payload["stream"] = req.stream
    return payload


async def _stream_upstream(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    # Generic byte-stream passthrough (SSE or chunked)
    async with client.stream("POST", url, headers=headers, json=json_payload) as resp:
        if resp.status_code < 200 or resp.status_code >= 300:
            text_preview = (await resp.aread()).decode("utf-8", errors="ignore")
            logger.error(f"[Seedream] Non-2xx {resp.status_code}, body: {text_preview[:1000]}")
            raise HTTPException(status_code=resp.status_code, detail=text_preview[:1000])
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            yield chunk


@router.post(
    "/doubao/v3/images/generations",
    summary="Proxy Doubao Seedream 4.0 image generation",
    tags=["AI Proxy"],
)
async def seedream_generate(payload: Dict[str, Any] = Body(...), request: Request = None):
    """
    Doubao Seedream 4.0 proxy:
    - Accepts upstream-compatible fields plus apiAddress/apiKey
    - Supports non-stream and stream passthrough
    """
    try:
        req = DoubaoSeedreamRequest(**payload)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid Seedream request", "errors": e.errors()},
        )

    url = _build_upstream_url(req.apiAddress)
    headers = {
        "Authorization": f"Bearer {req.apiKey}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "EzTalkProxy-Seedream/1.0",
    }
    upstream_payload = _make_payload(req)

    # Prefer shared http client from app.state if available
    client = getattr(getattr(request, "app", None), "state", None)
    http_client: Optional[httpx.AsyncClient] = None
    if client and isinstance(getattr(client, "http_client", None), httpx.AsyncClient):
        http_client = client.http_client

    try:
        if upstream_payload.get("stream", False):
            # Streaming passthrough
            use_client = http_client or httpx.AsyncClient(
                timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True
            )
            gen = _stream_upstream(use_client, url, headers, upstream_payload)
            # Content type: upstream may use SSE (text/event-stream) or chunked JSON
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-stream request
        if http_client:
            resp = await http_client.post(url, headers=headers, json=upstream_payload)
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), http2=True, follow_redirects=True) as client2:
                resp = await client2.post(url, headers=headers, json=upstream_payload)

        content_type = resp.headers.get("Content-Type", "application/json")
        text_preview = resp.text[:1000] if resp.text else "(empty)"
        logger.info(f"[Seedream] Upstream {resp.status_code}, content-type={content_type}, preview={text_preview}")

        if resp.status_code < 200 or resp.status_code >= 300:
            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": {
                        "message": text_preview,
                        "status": resp.status_code,
                    }
                },
            )

        # Forward upstream JSON as-is (no normalization to internal schema)
        try:
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception:
            # Non-JSON body fallback
            return JSONResponse(
                status_code=resp.status_code,
                content={"raw": resp.text},
            )

    except httpx.RequestError as e:
        logger.error(f"[Seedream] Request error to {url}: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Upstream request error: {e}")
    except HTTPException:
        # Already structured; just bubble up
        raise
    except Exception as e:
        logger.error(f"[Seedream] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")