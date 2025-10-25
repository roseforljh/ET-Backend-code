import logging
from typing import Dict, Any, Optional, AsyncGenerator

from ...models.api_models import AppStreamEventPy
from ...utils.helpers import get_current_time_iso, orjson_dumps_bytes_wrapper, to_sse_bytes

logger = logging.getLogger("EzTalkProxy.StreamProcessors.Cleanup")

async def handle_stream_cleanup(
    processing_state: Dict[str, Any], request_id: str,
    upstream_ok: bool, use_old_custom_separator_logic: bool, provider: str
) -> AsyncGenerator[bytes, None]:
    log_prefix = f"RID-{request_id}"
    state = processing_state
    logger.info(f"{log_prefix}: Stream cleanup. Provider: {provider}. Upstream OK: {upstream_ok}. CustomSep: {use_old_custom_separator_logic}")

    had_any_reasoning = state.get("had_any_reasoning", False)
    reasoning_finish_sent = state.get("reasoning_finish_event_sent", False)

    if had_any_reasoning and not reasoning_finish_sent:
        logger.info(f"{log_prefix}: Cleanup: Sending reasoning_finish event.")
        yield to_sse_bytes(AppStreamEventPy(type="reasoning_finish", timestamp=get_current_time_iso()))

    if not state.get("final_finish_event_sent_by_llm_reason"):
        final_reason = "stream_end"
        if not upstream_ok:
            final_reason = "upstream_error_or_connection_failed"
        
        logger.info(f"{log_prefix}: Cleanup: Sending final finish event with reason '{final_reason}'.")
        yield to_sse_bytes(AppStreamEventPy(type="finish", reason=final_reason, timestamp=get_current_time_iso()))