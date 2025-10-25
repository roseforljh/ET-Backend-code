import logging
import asyncio
import httpx
from typing import AsyncGenerator, Optional

from ...models.api_models import AppStreamEventPy
from ...utils.helpers import get_current_time_iso, orjson_dumps_bytes_wrapper, to_sse_bytes

logger = logging.getLogger("EzTalkProxy.StreamProcessors.ErrorHandling")

async def handle_stream_error(
    error: Exception, 
    request_id: str, 
    upstream_responded_ok: bool, 
    first_chunk_from_llm_received: bool,
    custom_message: Optional[str] = None
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    logger.error(f"{log_prefix}: Stream error: {type(error).__name__} - {error}", exc_info=True)
    
    # 如果提供了自定义消息，优先使用
    if custom_message:
        error_message = custom_message
        logger.info(f"{log_prefix}: Using custom error message: {error_message}")
    else:
        # 否则根据异常类型生成错误消息（保留原有逻辑）
        error_message = f"Stream processing error: {type(error).__name__}"
        
        if isinstance(error, httpx.TimeoutException): 
            error_message = "Request to LLM API timed out."
        elif isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            if status_code == 429:
                error_message = "请求频率过高 (429 Too Many Requests)，请稍后重试。服务器暂时限制了请求频率。"
            elif status_code == 401:
                error_message = "身份验证失败 (401 Unauthorized)，请检查API密钥配置。"
            elif status_code == 403:
                error_message = "访问被拒绝 (403 Forbidden)，请检查权限设置或API配额。"
            elif status_code == 404:
                error_message = "服务端点未找到 (404 Not Found)，请检查API地址配置。"
            elif status_code >= 500:
                error_message = f"服务器内部错误 ({status_code})，请稍后重试。"
            else:
                error_message = f"HTTP错误 {status_code}: {error.response.reason_phrase or 'Unknown error'}"
        elif isinstance(error, httpx.RequestError): 
            error_message = f"Network error: {error}"
        elif isinstance(error, asyncio.CancelledError): 
            logger.info(f"{log_prefix}: Stream cancelled.")
            return
        else: 
            error_message = f"Unexpected error: {str(error)[:200]}"
    
    yield to_sse_bytes(AppStreamEventPy(type="error", message=error_message, timestamp=get_current_time_iso()))
    yield to_sse_bytes(AppStreamEventPy(type="finish", reason="error_in_stream", timestamp=get_current_time_iso()))