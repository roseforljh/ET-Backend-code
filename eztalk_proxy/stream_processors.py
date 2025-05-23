import logging
import os
import orjson
import httpx 
import asyncio 
from typing import Dict, Any, AsyncGenerator, List

from .models import ChatRequest
from .utils import get_current_time_iso, strip_potentially_harmful_html_and_normalize_newlines, orjson_dumps_bytes_wrapper
from .config import THINKING_PROCESS_SEPARATOR, MIN_FLUSH_LENGTH_HEURISTIC

logger = logging.getLogger("EzTalkProxy.StreamProcessors")

def should_apply_custom_separator_logic(
    rd: ChatRequest, 
    request_id: str, 
    is_google_like_path_in_use: bool, 
    is_native_thinking_active: bool
) -> bool:
    if is_google_like_path_in_use and is_native_thinking_active:
        logger.info(f"RID-{request_id}: Native thinking active for Google-like path, custom separator logic OFF.")
        return False

    if rd.force_custom_reasoning_prompt is True:
        logger.info(f"RID-{request_id}: Custom separator logic FORCED for model '{rd.model}' by client.")
        return True
    
    logger.info(f"RID-{request_id}: Custom separator logic OFF by default for model '{rd.model}' (not forced and native thinking not conflicting).")
    return False


async def process_openai_response(parsed_sse_data: Dict[str, Any], state: Dict[str, Any], request_id: str) -> AsyncGenerator[bytes, None]:
    logger.debug(f"RID-{request_id}: process_openai_response received: {parsed_sse_data}") 
    
    state.setdefault("accumulated_openai_content", "")
    state.setdefault("accumulated_openai_reasoning", "")
    state.setdefault("openai_had_any_reasoning", False)
    state.setdefault("openai_had_any_content_or_tool_call", False)
    state.setdefault("openai_reasoning_finish_event_sent", False)


    for choice in parsed_sse_data.get('choices', []):
        delta = choice.get('delta', {})
        finish_reason = choice.get("finish_reason")

        reasoning_chunk_raw = delta.get("reasoning_content") 
        content_chunk_raw = delta.get("content")
        tool_calls_chunk = delta.get("tool_calls")

        if reasoning_chunk_raw is not None:
            if not isinstance(reasoning_chunk_raw, str): reasoning_chunk_raw = str(reasoning_chunk_raw) 
            state["accumulated_openai_reasoning"] += reasoning_chunk_raw
            state["openai_had_any_reasoning"] = True 
            logger.debug(f"RID-{request_id}: OpenAI: Accumulated reasoning, new total length: {len(state['accumulated_openai_reasoning'])}")

        if content_chunk_raw is not None:
            if not isinstance(content_chunk_raw, str): content_chunk_raw = str(content_chunk_raw) 
            state["accumulated_openai_content"] += content_chunk_raw
            state["openai_had_any_content_or_tool_call"] = True
            logger.debug(f"RID-{request_id}: OpenAI: Accumulated content, new total length: {len(state['accumulated_openai_content'])}")
        
        should_flush_reasoning = False
        should_flush_content = False

        if tool_calls_chunk or finish_reason: 
            should_flush_reasoning = True
            should_flush_content = True
        elif state["accumulated_openai_reasoning"] and \
             ("\n\n" in state["accumulated_openai_reasoning"] or len(state["accumulated_openai_reasoning"]) >= MIN_FLUSH_LENGTH_HEURISTIC):
            should_flush_reasoning = True
        elif state["accumulated_openai_content"] and \
             ("\n\n" in state["accumulated_openai_content"] or len(state["accumulated_openai_content"]) >= MIN_FLUSH_LENGTH_HEURISTIC):
            should_flush_content = True

        if should_flush_reasoning and state["accumulated_openai_reasoning"]:
            text_to_process = state["accumulated_openai_reasoning"]
            chunk_to_yield = ""
            if not (tool_calls_chunk or finish_reason) and "\n\n" in text_to_process: 
                split_at = text_to_process.rfind("\n\n") + 2 
                chunk_to_yield = text_to_process[:split_at]
                state["accumulated_openai_reasoning"] = text_to_process[split_at:]
            else: 
                chunk_to_yield = text_to_process
                state["accumulated_openai_reasoning"] = ""
            
            if chunk_to_yield:
                processed_text = strip_potentially_harmful_html_and_normalize_newlines(chunk_to_yield)
                if processed_text:
                    logger.debug(f"RID-{request_id}: Yielding OpenAI reasoning (buffered): '{processed_text[:100]}'")
                    yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_text, "timestamp": get_current_time_iso()})
        
        if (should_flush_content or tool_calls_chunk or finish_reason) and \
           state.get("openai_had_any_reasoning") and \
           not state.get("openai_reasoning_finish_event_sent"):
            if state["accumulated_openai_reasoning"]: 
                processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_openai_reasoning"])
                if processed_text: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_text, "timestamp": get_current_time_iso()})
                state["accumulated_openai_reasoning"] = ""
            logger.debug(f"RID-{request_id}: Yielding OpenAI reasoning_finish.")
            yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
            state["openai_reasoning_finish_event_sent"] = True
        
        if should_flush_content and state["accumulated_openai_content"]:
            text_to_process = state["accumulated_openai_content"]
            chunk_to_yield = ""
            if not (tool_calls_chunk or finish_reason) and "\n\n" in text_to_process: 
                split_at = text_to_process.rfind("\n\n") + 2
                chunk_to_yield = text_to_process[:split_at]
                state["accumulated_openai_content"] = text_to_process[split_at:]
            else: 
                chunk_to_yield = text_to_process
                state["accumulated_openai_content"] = ""
            
            if chunk_to_yield:
                processed_text = strip_potentially_harmful_html_and_normalize_newlines(chunk_to_yield)
                if processed_text:
                    logger.debug(f"RID-{request_id}: Yielding OpenAI content (buffered): '{processed_text[:100]}'")
                    yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_text, "timestamp": get_current_time_iso()})

        if tool_calls_chunk:
            logger.debug(f"RID-{request_id}: Yielding OpenAI tool_calls_chunk: {tool_calls_chunk}")
            yield orjson_dumps_bytes_wrapper({"type": "tool_calls_chunk", "data": tool_calls_chunk, "timestamp": get_current_time_iso()})
            state["openai_had_any_content_or_tool_call"] = True 

        if finish_reason:
            logger.info(f"RID-{request_id}: OpenAI choice finish_reason: {finish_reason}.")
            if state.get("openai_had_any_reasoning") and not state.get("openai_reasoning_finish_event_sent"):
                logger.debug(f"RID-{request_id}: OpenAI finish_reason triggered final reasoning_finish.")
                yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                state["openai_reasoning_finish_event_sent"] = True


async def process_google_response(
    parsed_sse_data: Dict[str, Any], 
    state: Dict[str, Any], 
    request_id: str, 
    is_native_thinking_active: bool,
    use_old_custom_separator_logic_active: bool 
) -> AsyncGenerator[bytes, None]:
    logger.debug(f"RID-{request_id}: process_google_response. NativeThinking={is_native_thinking_active}, CustomSeparatorLogic={use_old_custom_separator_logic_active}. Data: {parsed_sse_data}")
    
    state.setdefault("accumulated_google_thought", "")
    state.setdefault("accumulated_google_text", "")
    state.setdefault("google_native_had_thoughts", False)
    state.setdefault("google_native_had_answer", False) 
    state.setdefault("openai_reasoning_finish_event_sent", False) 

    state.setdefault("accumulated_text_custom", "")
    state.setdefault("full_yielded_reasoning_custom", "")
    state.setdefault("full_yielded_content_custom", "")
    state.setdefault("found_separator_custom", False)


    for candidate_idx, candidate in enumerate(parsed_sse_data.get('candidates', [])):
        content = candidate.get("content", {})
        finish_reason = candidate.get("finishReason") 
        
        if not content.get("parts") and not finish_reason: 
            logger.debug(f"RID-{request_id}: Google Candidate {candidate_idx} has no parts and no finish_reason. Skipping.")
            continue

        candidate_had_any_output_this_chunk = False

        if is_native_thinking_active:
            for part_idx, part in enumerate(content.get("parts", [])):
                is_thought_part = part.get("thought", False) is True 
                text_from_part = part.get("text")
                function_call_from_part = part.get("functionCall")

                if text_from_part is not None:
                    candidate_had_any_output_this_chunk = True
                    if not isinstance(text_from_part, str): text_from_part = str(text_from_part)
                    
                    if is_thought_part: 
                        state["accumulated_google_thought"] += text_from_part
                        state['google_native_had_thoughts'] = True
                        logger.debug(f"RID-{request_id}: Google (Native): Accumulated thought, new total length: {len(state['accumulated_google_thought'])}")
                    else: 
                        state["accumulated_google_text"] += text_from_part
                        state['google_native_had_answer'] = True 
                        logger.debug(f"RID-{request_id}: Google (Native): Accumulated text, new total length: {len(state['accumulated_google_text'])}")
                
                elif function_call_from_part: 
                    candidate_had_any_output_this_chunk = True
                    if state["accumulated_google_thought"]:
                        processed_thought = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_thought"])
                        if processed_thought: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_thought, "timestamp": get_current_time_iso()})
                        state["accumulated_google_thought"] = ""
                    
                    if state.get('google_native_had_thoughts') and not state.get('openai_reasoning_finish_event_sent'):
                        yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                        state['openai_reasoning_finish_event_sent'] = True 
                    
                    fcid = f"gemini_fc_native_{os.urandom(4).hex()}" 
                    yield orjson_dumps_bytes_wrapper({
                        "type": "google_function_call_request", 
                        "id": fcid,
                        "name": function_call_from_part.get("name"),
                        "arguments_obj": function_call_from_part.get("args", {}), 
                        "timestamp": get_current_time_iso(),
                        "is_reasoning_step": False 
                    })
                    state['google_native_had_answer'] = True 
            
            should_flush_google_thought = False
            should_flush_google_text = False

            if finish_reason: 
                should_flush_google_thought = True
                should_flush_google_text = True
            elif state["accumulated_google_thought"] and \
                 ("\n\n" in state["accumulated_google_thought"] or len(state["accumulated_google_thought"]) >= MIN_FLUSH_LENGTH_HEURISTIC):
                should_flush_google_thought = True
            elif state["accumulated_google_text"] and \
                 ("\n\n" in state["accumulated_google_text"] or len(state["accumulated_google_text"]) >= MIN_FLUSH_LENGTH_HEURISTIC):
                should_flush_google_text = True
            
            if should_flush_google_thought and state["accumulated_google_thought"]:
                text_to_process = state["accumulated_google_thought"]
                chunk_to_yield = ""
                if not finish_reason and "\n\n" in text_to_process: 
                    split_at = text_to_process.rfind("\n\n") + 2
                    chunk_to_yield = text_to_process[:split_at]
                    state["accumulated_google_thought"] = text_to_process[split_at:]
                else: 
                    chunk_to_yield = text_to_process
                    state["accumulated_google_thought"] = ""
                
                if chunk_to_yield:
                    processed_text = strip_potentially_harmful_html_and_normalize_newlines(chunk_to_yield)
                    if processed_text: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_text, "timestamp": get_current_time_iso()})

            if (should_flush_google_text or finish_reason or (candidate_had_any_output_this_chunk and function_call_from_part)) and \
               state.get('google_native_had_thoughts') and \
               not state.get('openai_reasoning_finish_event_sent'):
                if state["accumulated_google_thought"]: 
                    processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_thought"])
                    if processed_text: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_text, "timestamp": get_current_time_iso()})
                    state["accumulated_google_thought"] = ""
                yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                state['openai_reasoning_finish_event_sent'] = True

            if should_flush_google_text and state["accumulated_google_text"]:
                text_to_process = state["accumulated_google_text"]
                chunk_to_yield = ""
                if not finish_reason and "\n\n" in text_to_process: 
                    split_at = text_to_process.rfind("\n\n") + 2
                    chunk_to_yield = text_to_process[:split_at]
                    state["accumulated_google_text"] = text_to_process[split_at:]
                else: 
                    chunk_to_yield = text_to_process
                    state["accumulated_google_text"] = ""
                
                if chunk_to_yield:
                    processed_text = strip_potentially_harmful_html_and_normalize_newlines(chunk_to_yield)
                    if processed_text: yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_text, "timestamp": get_current_time_iso()})
        
        else: 
            current_candidate_text = "".join(p.get("text","") for p in content.get("parts", []) if p.get("text") is not None)
            if current_candidate_text:
                candidate_had_any_output_this_chunk = True
                cleaned_text_chunk = strip_potentially_harmful_html_and_normalize_newlines(current_candidate_text)

                if cleaned_text_chunk:
                    if use_old_custom_separator_logic_active: 
                        state["accumulated_text_custom"] = state.get("accumulated_text_custom", "") + cleaned_text_chunk
                        logger.debug(f"RID-{request_id}: Appended to custom_accumulator (Google non-native): {cleaned_text_chunk[:100]}")
                    else: 
                        yield orjson_dumps_bytes_wrapper({"type": "content", "text": cleaned_text_chunk, "timestamp": get_current_time_iso()})
            
            for part in content.get("parts", []):
                if part.get("functionCall"):
                    candidate_had_any_output_this_chunk = True
                    fc = part["functionCall"]
                    fcid = f"gemini_fc_std_{os.urandom(4).hex()}"
                    yield orjson_dumps_bytes_wrapper({
                        "type": "google_function_call_request",
                        "id": fcid,
                        "name": fc.get("name"),
                        "arguments_obj": fc.get("args", {}),
                        "timestamp": get_current_time_iso(),
                        "is_reasoning_step": False 
                    })
        
        if not candidate_had_any_output_this_chunk and not finish_reason:
             logger.debug(f"RID-{request_id}: Google Candidate {candidate_idx} had no new output this chunk and no finish_reason.")


        if finish_reason:
            logger.info(f"RID-{request_id}: Google Candidate {candidate_idx} finish_reason: {finish_reason}.")
            if is_native_thinking_active and state.get('google_native_had_thoughts') and \
               not state.get('google_native_had_answer') and \
               not state.get('openai_reasoning_finish_event_sent'):
                if state["accumulated_google_thought"]: 
                    processed_text = strip_potentially_harmful_html_and_normalize_newlines(state["accumulated_google_thought"])
                    if processed_text: yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": processed_text, "timestamp": get_current_time_iso()})
                    state["accumulated_google_thought"] = ""
                logger.debug(f"RID-{request_id}: Google finish_reason, native thinking had thoughts but no answer. Sending reasoning_finish.")
                yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                state["openai_reasoning_finish_event_sent"] = True 
            
            yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": finish_reason, "timestamp": get_current_time_iso()})
            return 


async def handle_stream_error(e: Exception, request_id: str, upstream_ok: bool, first_chunk_llm: bool) -> AsyncGenerator[bytes, None]:
    err_type = "internal_server_error"
    err_msg = f"Internal server error: {str(e)}"
    log_exc_info = True

    if isinstance(e, httpx.TimeoutException):
        err_type, err_msg, log_exc_info = "timeout_error", f"Upstream timeout: {str(e)}", False
    elif isinstance(e, httpx.RequestError): 
        err_type, err_msg, log_exc_info = "network_error", f"Upstream network error: {str(e)}", False
    elif isinstance(e, asyncio.CancelledError):
        logger.info(f"RID-{request_id}: Stream cancelled by client.")
        return 
    
    logger.error(f"RID-{request_id}: Error during stream: {err_msg}", exc_info=log_exc_info)
    
    if not upstream_ok or not first_chunk_llm or err_type != "internal_server_error":
        yield orjson_dumps_bytes_wrapper({"type": "error", "message": err_msg, "timestamp": get_current_time_iso()})
    
    yield orjson_dumps_bytes_wrapper({"type": "finish", "reason": err_type, "timestamp": get_current_time_iso()})


async def handle_stream_cleanup(
    state: Dict[str, Any], 
    request_id: str, 
    upstream_ok: bool,
    use_old_custom_separator_branch: bool, 
    provider_for_log: str 
    ) -> AsyncGenerator[bytes, None]:

    if use_old_custom_separator_branch and state.get("accumulated_text_custom","").strip() and upstream_ok:
        logger.info(f"RID-{request_id}: Custom Separator Cleanup: Processing accumulated: '{state.get('accumulated_text_custom','')[:100]}'")
        
        final_raw_text_custom = state.get("accumulated_text_custom","")
        
        if not state.get("found_separator_custom"): 
            separator_index = final_raw_text_custom.find(THINKING_PROCESS_SEPARATOR)
            if separator_index != -1:
                state["found_separator_custom"] = True
                reasoning_part = final_raw_text_custom[:separator_index]
                content_part = final_raw_text_custom[separator_index + len(THINKING_PROCESS_SEPARATOR):]

                processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(reasoning_part)
                delta_reasoning = processed_reasoning[len(state.get("full_yielded_reasoning_custom", "")):]
                if delta_reasoning: 
                    yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": delta_reasoning, "timestamp": get_current_time_iso()})
                    state["full_yielded_reasoning_custom"] = state.get("full_yielded_reasoning_custom", "") + delta_reasoning
                
                if not state.get("openai_reasoning_finish_event_sent"): 
                    yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                    state["openai_reasoning_finish_event_sent"] = True 
                
                processed_content = strip_potentially_harmful_html_and_normalize_newlines(content_part)
                if processed_content: 
                    yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_content, "timestamp": get_current_time_iso()})
                    state["full_yielded_content_custom"] = state.get("full_yielded_content_custom", "") + processed_content
            
            else: 
                processed_reasoning = strip_potentially_harmful_html_and_normalize_newlines(final_raw_text_custom)
                delta_reasoning = processed_reasoning[len(state.get("full_yielded_reasoning_custom", "")):]
                if delta_reasoning: 
                    yield orjson_dumps_bytes_wrapper({"type": "reasoning", "text": delta_reasoning, "timestamp": get_current_time_iso()})
                    state["full_yielded_reasoning_custom"] = state.get("full_yielded_reasoning_custom", "") + delta_reasoning
                
                if state.get("full_yielded_reasoning_custom") and not state.get("openai_reasoning_finish_event_sent"):
                    yield orjson_dumps_bytes_wrapper({"type": "reasoning_finish", "timestamp": get_current_time_iso()})
                    state["openai_reasoning_finish_event_sent"] = True
        
        elif state.get("found_separator_custom"): 
            current_total_yielded_len = len(state.get("full_yielded_reasoning_custom","")) + \
                                        len(THINKING_PROCESS_SEPARATOR) + \
                                        len(state.get("full_yielded_content_custom",""))
            
            remaining_custom_text_content = final_raw_text_custom[current_total_yielded_len:]

            if remaining_custom_text_content.strip():
                processed_content = strip_potentially_harmful_html_and_normalize_newlines(remaining_custom_text_content)
                if processed_content:
                    yield orjson_dumps_bytes_wrapper({"type": "content", "text": processed_content, "timestamp": get_current_time_iso()})

    final_processing_mode_info = []
    if use_old_custom_separator_branch: 
        final_processing_mode_info.append("OldCustomSeparatorActive")
        if state.get("found_separator_custom"): final_processing_mode_info.append("SeparatorFound")
        if state.get("full_yielded_reasoning_custom"): final_processing_mode_info.append("YieldedCustomReasoning")
        if state.get("full_yielded_content_custom"): final_processing_mode_info.append("YieldedCustomContent")
    
    is_final_log_google_native_path = state.get("_is_native_thinking_final_log", False) 

    if provider_for_log == "google": 
        if is_final_log_google_native_path: 
            final_processing_mode_info.append("GoogleNativeThinkingUsed")
            if state.get('google_native_had_thoughts'): final_processing_mode_info.append("HadNativeThoughts")
            if state.get('google_native_had_answer'): final_processing_mode_info.append("HadNativeAnswer")
        elif not use_old_custom_separator_branch: 
            if state.get("accumulated_google_text") or any(event.get("type")=="content" for event in state.get("_events_yielded",[])): 
                 final_processing_mode_info.append("GoogleDirectContent")
    elif provider_for_log == "openai":
        if not use_old_custom_separator_branch: 
            if state.get('openai_had_any_reasoning'): final_processing_mode_info.append("OpenAIWithReasoningField")
            if state.get('openai_had_any_content_or_tool_call'): final_processing_mode_info.append("OpenAINormalContentOrToolCall")
    
    if not final_processing_mode_info:
        final_processing_mode_info.append("StreamEndedEarlyOrNoMatchingOutputState")
        
    logger.info(f"RID-{request_id}: Stream cleanup. Provider: {provider_for_log}. Modes: {', '.join(final_processing_mode_info)}.")