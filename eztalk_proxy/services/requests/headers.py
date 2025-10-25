"""
Headers builders for request construction.

During migration, these helpers mirror the legacy behavior in
services.request_builder while providing a stable place for future changes.
"""

from __future__ import annotations

from typing import Dict, Any


def build_openai_headers(request_data: Any) -> Dict[str, str]:
    """
    Build headers for OpenAI-compatible endpoints.
    Mirrors the legacy behavior:
      - Authorization: Bearer <api_key>
      - Content-Type: application/json
      - Accept: text/event-stream
      - x-api-key: <api_key>
      - x-goog-api-key: <api_key> (only if model name contains 'gemini' in OpenAI format)
    
    Additionally adds browser-like headers to avoid Cloudflare blocking.
    """
    api_key = getattr(request_data, "api_key", "") or ""
    model = (getattr(request_data, "model", "") or "").lower()

    # 不设置 Accept-Encoding，让 httpx 自动处理压缩
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream,application/json,*/*",
        "x-api-key": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
    }

    # Some aggregators require x-goog-api-key when calling Gemini via OpenAI format
    if "gemini" in model:
        headers["x-goog-api-key"] = api_key

    return headers


def build_gemini_headers(request_data: Any) -> Dict[str, str]:
    """
    Build headers for Gemini REST API endpoints.
    Mirrors legacy behavior:
      - Content-Type: application/json
      - x-goog-api-key: <api_key>
    """
    api_key = getattr(request_data, "api_key", "") or ""
    return {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }