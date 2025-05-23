import os
import logging
import httpx
import orjson
from typing import Optional, Dict, Any, AsyncGenerator, List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..models import ChatRequest, ApiMessage
from ..config import (
    GOOGLE_API_BASE_URL, COMMON_HEADERS # <--- 确保 COMMON_HEADERS 在这里
)
from ..utils import (
    # error_response, # 这个主要用于非流式错误，在流式中需要特殊处理
    extract_sse_lines, get_current_time_iso,
    orjson_dumps_bytes_wrapper, strip_potentially_harmful_html_and_normalize_newlines
)
from ..api_helpers import prepare_openai_request, prepare_google_request_payload_structure
from ..web_search import perform_web_search, generate_search_context_message_content
from ..stream_processors import (
    should_apply_custom_separator_logic, process_openai_response, process_google_response,
    handle_stream_error, handle_stream_cleanup
)

logger = logging.getLogger("EzTalkProxy.Routers.Chat")
router = APIRouter()

async def get_http_client(request: Request) -> Optional[httpx.AsyncClient]:
    return getattr(request.app.state, "http_client", None)

@router.post("/chat", response_class=StreamingResponse, summary="AI聊天完成代理", tags=["AI Proxy"])
async def chat_proxy(
    request_data: ChatRequest,
    client: Optional[httpx.AsyncClient] = Depends(get_http_client)
):
    request_id = os.urandom(8).hex()
    logger.info(
        f"RID-{request_id}: Received /chat request: Provider='{request_data.provider}', "
        f"Model='{request_data.model}', WebSearch={request_data.use_web_search}, "
        f"ForceCustomReasoning={request_data.force_custom_reasoning_prompt}, "
        f"CustomParams={request_data.custom_model_parameters is not None}"
    )

    if not client:
        logger.warning(f"RID-{request_id}: HTTP client not available from app.state for request {request_id}.")
        async def client_error_gen():
            yield orjson_dumps_bytes_wrapper({"type": "error", "message": "Service unavailable: HTTP client not initialized.", "timestamp": get_current_time_iso()})
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "service_unavailable", "timestamp": get_current_time_iso()})
        return StreamingResponse(client_error_gen(), media_type="text/event-stream", headers=COMMON_HEADERS)


    api_messages_for_processing: List[ApiMessage] = [
        m.model_copy(deep=True) for m in request_data.messages
        if m.content is not None or m.tool_calls is not None or m.role == "system"
    ]

    if not any(m.role != "system" for m in api_messages_for_processing):
         if not any(m.role == "system" and m.content for m in api_messages_for_processing):
            async def no_message_error_gen():
                yield orjson_dumps_bytes_wrapper({"type": "error", "message": "No processable messages provided (excluding empty system messages).", "timestamp": get_current_time_iso()})
                yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "bad_request", "timestamp": get_current_time_iso()})
            return StreamingResponse(no_message_error_gen(), media_type="text/event-stream", headers=COMMON_HEADERS) # COMMON_HEADERS 被使用

    user_query_for_search = ""
    search_results_generated = False
    is_native_thinking_mode_active = False
    use_google_sse_parser_flag = False
    is_google_payload_format_used_flag = False
    is_google_like_path_active = False

    if request_data.use_web_search:
        for msg_obj in reversed(api_messages_for_processing):
            if msg_obj.role == "user" and msg_obj.content and msg_obj.content.strip():
                user_query_for_search = msg_obj.content.strip()
                break
        if not user_query_for_search:
            logger.warning(f"RID-{request_id}: Web search enabled but no user query found in messages.")

    if request_data.provider not in ["google", "openai"]:
        async def provider_error_gen():
            yield orjson_dumps_bytes_wrapper({"type": "error", "message": f"Unsupported provider: {request_data.provider}", "timestamp": get_current_time_iso()})
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "bad_request", "timestamp": get_current_time_iso()})
        return StreamingResponse(provider_error_gen(), media_type="text/event-stream", headers=COMMON_HEADERS) # COMMON_HEADERS 被使用
    elif request_data.provider == "google":
        logger.info(f"RID-{request_id}: Path: Google Direct. Model: {request_data.model}")
        is_google_payload_format_used_flag = True
        use_google_sse_parser_flag = True
        is_google_like_path_active = True
    elif request_data.provider == "openai":
        logger.info(f"RID-{request_id}: Path: OpenAI Provider ('{request_data.model}').")

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        nonlocal api_messages_for_processing, search_results_generated, user_query_for_search
        nonlocal is_native_thinking_mode_active, use_google_sse_parser_flag
        nonlocal is_google_payload_format_used_flag, is_google_like_path_active

        if not client:
            logger.error(f"RID-{request_id}: CRITICAL - HTTP client is None within stream_generator scope.")
            yield orjson_dumps_bytes_wrapper({"type": "error", "message": "Internal error: HTTP client unavailable in stream.", "timestamp": get_current_time_iso()})
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "internal_error", "timestamp": get_current_time_iso()})
            return
        
        # ... (stream_generator 的其余内容保持不变) ...
        try:
            if request_data.use_web_search and user_query_for_search:
                yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_started", "timestamp": get_current_time_iso()})
                search_results_list = await perform_web_search(user_query_for_search, request_id)
                if search_results_list:
                    search_context_content = generate_search_context_message_content(user_query_for_search, search_results_list)
                    new_system_message = ApiMessage(role="system", content=search_context_content)
                    last_user_message_index = -1
                    for i, msg in reversed(list(enumerate(api_messages_for_processing))):
                        if msg.role == "user":
                            last_user_message_index = i
                            break
                    if last_user_message_index != -1:
                        api_messages_for_processing.insert(last_user_message_index, new_system_message)
                    else:
                        api_messages_for_processing.append(new_system_message)
                    search_results_generated = True
                    logger.info(f"RID-{request_id}: Web search context injected into messages.")
                    yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_complete_with_results", "query": user_query_for_search, "timestamp": get_current_time_iso()})
                else:
                    logger.info(f"RID-{request_id}: Web search yielded no results for query '{user_query_for_search[:100]}'.")
                    yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_search_complete_no_results", "query": user_query_for_search, "timestamp": get_current_time_iso()})
                yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_analysis_started", "timestamp": get_current_time_iso()})

            current_api_url: str; current_api_headers: Dict[str,str]; current_api_payload: Dict[str,Any]; current_api_params: Optional[Dict[str,str]] = None

            if request_data.provider == "google":
                payload_dict, native_thinking_flag = prepare_google_request_payload_structure(
                    request_data, api_messages_for_processing, request_id
                )
                is_native_thinking_mode_active = native_thinking_flag
                current_api_payload = payload_dict
                current_api_url = f"{GOOGLE_API_BASE_URL}/v1beta/models/{request_data.model}:streamGenerateContent"
                current_api_params = {"key": request_data.api_key, "alt": "sse"}
                current_api_headers = {"Content-Type": "application/json"}
            elif request_data.provider == "openai":
                current_api_url, current_api_headers, current_api_payload = prepare_openai_request(
                    request_data, api_messages_for_processing, request_id
                )

            use_old_custom_separator_branch_flag = should_apply_custom_separator_logic(
                request_data, request_id,
                is_google_like_path_active,
                is_native_thinking_mode_active
            )
            logger.info(f"RID-{request_id}: Final Logic Flags: GooglePayloadUsed={is_google_payload_format_used_flag}, ExpectGoogleResponseSSE={use_google_sse_parser_flag}, NativeThinkingActiveForGooglePath={is_native_thinking_mode_active}, UseOldSeparatorLogic={use_old_custom_separator_branch_flag}")
            logger.debug(f"RID-{request_id}: Sending to URL: {current_api_url}. Headers: {current_api_headers}. Payload (first 500 of messages/contents): {str(current_api_payload.get('messages', current_api_payload.get('contents',[])))[:500]}")

            buffer = bytearray()
            upstream_ok = False
            first_chunk_llm = False
            state: Dict[str, Any] = {
                "accumulated_openai_content": "", "accumulated_openai_reasoning": "",
                "openai_had_any_reasoning": False, "openai_had_any_content_or_tool_call": False,
                "openai_reasoning_finish_event_sent": False,
                "accumulated_google_thought": "", "accumulated_google_text": "",
                "google_native_had_thoughts": False, "google_native_had_answer": False,
                "accumulated_text_custom": "", "full_yielded_reasoning_custom": "",
                "full_yielded_content_custom": "", "found_separator_custom": False,
            }

            async with client.stream("POST", current_api_url, headers=current_api_headers, json=current_api_payload, params=current_api_params) as resp:
                logger.info(f"RID-{request_id}: Upstream LLM response status: {resp.status_code}")
                if not (200 <= resp.status_code < 300):
                    err_body_bytes = await resp.aread()
                    err_text = err_body_bytes.decode("utf-8", errors="replace")
                    logger.error(f"RID-{request_id}: Upstream LLM error {resp.status_code}: {err_text[:1000]}")
                    try:
                        err_data = orjson.loads(err_text)
                        msg_detail = err_data.get("error", {}).get("message", str(err_data))
                    except:
                        msg_detail = err_text[:200]
                    yield orjson_dumps_bytes_wrapper({"type": "error", "message": f"LLM API Error: {msg_detail}", "upstream_status": resp.status_code, "timestamp": get_current_time_iso()})
                    yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "upstream_error", "timestamp": get_current_time_iso()})
                    return

                upstream_ok = True
                async for raw_chunk_bytes in resp.aiter_raw():
                    if not raw_chunk_bytes: continue
                    if not first_chunk_llm:
                        if request_data.use_web_search and user_query_for_search:
                            yield orjson_dumps_bytes_wrapper({"type": "status_update", "stage": "web_analysis_complete", "timestamp": get_current_time_iso()})
                        first_chunk_llm = True
                    buffer.extend(raw_chunk_bytes)
                    sse_lines, buffer = extract_sse_lines(buffer)

                    for sse_line_bytes in sse_lines:
                        if not sse_line_bytes.strip(): continue
                        sse_data_bytes = b""
                        if sse_line_bytes.startswith(b"data: "):
                            sse_data_bytes = sse_line_bytes[len(b"data: "):].strip()
                        if not sse_data_bytes: continue
                        logger.debug(f"RID-{request_id}, Raw SSE Data Line: {sse_data_bytes!r}")

                        if sse_data_bytes == b"[DONE]":
                            if not use_google_sse_parser_flag:
                                logger.info(f"RID-{request_id}: Received [DONE] from OpenAI-like endpoint.")
                                if state.get("accumulated_openai_reasoning"):
                                    processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_reasoning"])
                                    if processed_reasoning: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_reasoning, "timestamp": get_current_time_iso()})
                                    state["accumulated_openai_reasoning"] = ""
                                if state.get("openai_had_any_reasoning") and not state.get("openai_reasoning_finish_event_sent"):
                                    yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                                    state["openai_reasoning_finish_event_sent"] = True
                                if state.get("accumulated_openai_content"):
                                    processed_content = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_content"])
                                    if processed_content: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_content, "timestamp": get_current_time_iso()})
                                    state["accumulated_openai_content"] = ""
                                yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "stop_openai_done", "timestamp": get_current_time_iso()})
                            else:
                                logger.warning(f"RID-{request_id}: Received [DONE] but was expecting Google format SSE. Treating as end.")
                                if state.get("accumulated_google_thought"):
                                    processed_thought = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_thought"])
                                    if processed_thought: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_thought, "timestamp": get_current_time_iso()})
                                    state["accumulated_google_thought"] = ""
                                if is_native_thinking_mode_active and state.get('google_native_had_thoughts') and not state.get('google_native_had_answer') and not state.get('openai_reasoning_finish_event_sent'):
                                    yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                                    state["openai_reasoning_finish_event_sent"] = True
                                if state.get("accumulated_google_text"):
                                    processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_text"])
                                    if processed_text: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_text, "timestamp": get_current_time_iso()})
                                    state["accumulated_google_text"] = ""
                                yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": "google_stream_ended_with_unexpected_done_signal", "timestamp": get_current_time_iso()})
                            return

                        try:
                            parsed_sse_data = orjson.loads(sse_data_bytes)
                            logger.debug(f"RID-{request_id}, Parsed SSE Data JSON: {parsed_sse_data}")
                        except orjson.JSONDecodeError:
                            logger.warning(f"RID-{request_id}: Failed to parse SSE JSON data: {sse_data_bytes[:100]!r}"); continue

                        if use_google_sse_parser_flag:
                            async for event in process_google_response(parsed_sse_data, state, request_id, is_native_thinking_mode_active, use_old_custom_separator_branch_flag):
                                yield event
                                if event:
                                    try:
                                        event_data = orjson.loads(event)
                                        if event_data.get("type") == "finish": return
                                    except orjson.JSONDecodeError: pass
                        else:
                            async for event in process_openai_response(parsed_sse_data, state, request_id):
                                yield event
        except Exception as e:
            async for event in handle_stream_error(e, request_id, upstream_ok, first_chunk_llm):
                yield event
        finally:
            if upstream_ok :
                if not use_google_sse_parser_flag :
                    if state.get("accumulated_openai_reasoning"):
                        logger.info(f"RID-{request_id}: FINALLY flushing OpenAI reasoning: '{state['accumulated_openai_reasoning'][:100]}'")
                        processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_reasoning"])
                        if processed_reasoning: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_reasoning, "timestamp": get_current_time_iso()})
                    if state.get("openai_had_any_reasoning") and not state.get("openai_reasoning_finish_event_sent"):
                        logger.info(f"RID-{request_id}: FINALLY sending OpenAI reasoning_finish.")
                        yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                        state["openai_reasoning_finish_event_sent"] = True
                    if state.get("accumulated_openai_content"):
                        logger.info(f"RID-{request_id}: FINALLY flushing OpenAI content: '{state['accumulated_openai_content'][:100]}'")
                        processed_content = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_content"])
                        if processed_content: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_content, "timestamp": get_current_time_iso()})
                elif use_google_sse_parser_flag:
                    if state.get("accumulated_google_thought"):
                        logger.info(f"RID-{request_id}: FINALLY flushing Google thought: '{state['accumulated_google_thought'][:100]}'")
                        processed_thought = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_thought"])
                        if processed_thought: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_thought, "timestamp": get_current_time_iso()})
                    if is_native_thinking_mode_active and state.get('google_native_had_thoughts') and not state.get('openai_reasoning_finish_event_sent'):
                        logger.info(f"RID-{request_id}: FINALLY sending Google reasoning_finish (native thoughts).")
                        yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                        state["openai_reasoning_finish_event_sent"] = True
                    if state.get("accumulated_google_text"):
                        logger.info(f"RID-{request_id}: FINALLY flushing Google text: '{state['accumulated_google_text'][:100]}'")
                        processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_text"])
                        if processed_text: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_text, "timestamp": get_current_time_iso()})

            state["_is_native_thinking_final_log"] = is_native_thinking_mode_active if is_google_like_path_active else False
            async for event in handle_stream_cleanup(
                state, request_id, upstream_ok,
                use_old_custom_separator_branch_flag,
                request_data.provider
            ):
                yield event

    return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=COMMON_HEADERS)