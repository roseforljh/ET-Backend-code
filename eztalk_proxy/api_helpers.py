import logging
import orjson
from typing import List, Dict, Any, Optional, Union, Tuple

from .models import ApiMessage, ChatRequest
from .config import DEFAULT_OPENAI_API_BASE_URL, OPENAI_COMPATIBLE_PATH
from .katex_prompt import KATEX_FORMATTING_INSTRUCTION
from .utils import is_gemini_2_5_model

logger = logging.getLogger("EzTalkProxy.APIHelpers")

def _convert_openai_tools_to_gemini_declarations(openai_tools: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    declarations = []
    if not openai_tools:
        return []
    for tool_def in openai_tools:
        if tool_def.get("type") == "function" and "function" in tool_def:
            func_spec = tool_def["function"]
            declaration = {
                k: v for k, v in {
                    "name": func_spec.get("name"),
                    "description": func_spec.get("description"),
                    "parameters": func_spec.get("parameters")
                }.items() if v is not None
            }
            if declaration.get("name"):
                declarations.append(declaration)
            else:
                logger.warning(f"RID-{request_id}: Google tool conversion: Tool definition missing name: {func_spec}")
    return declarations

def _convert_openai_tool_choice_to_gemini_tool_config(
    openai_tool_choice: Union[str, Dict[str, Any]],
    gemini_declarations: List[Dict[str, Any]],
    request_id: str
) -> Optional[Dict[str, Any]]:
    if not openai_tool_choice:
        return None

    mode = "AUTO"
    allowed_function_names = []

    if isinstance(openai_tool_choice, str):
        if openai_tool_choice == "none":
            mode = "NONE"
        elif openai_tool_choice == "auto":
            mode = "AUTO"
        elif openai_tool_choice == "required":
            mode = "ANY" if gemini_declarations else "AUTO"
        else:
            logger.warning(f"RID-{request_id}: Google tool_choice: Unsupported str value '{openai_tool_choice}', defaulting to AUTO.")
            mode = "AUTO"
    elif isinstance(openai_tool_choice, dict) and openai_tool_choice.get("type") == "function":
        func_name = openai_tool_choice.get("function", {}).get("name")
        if func_name:
            if any(decl["name"] == func_name for decl in gemini_declarations):
                mode = "ANY"
                allowed_function_names = [func_name]
            else:
                logger.warning(f"RID-{request_id}: Google tool_choice: Specified function '{func_name}' not in declared tools. Defaulting to AUTO.")
                mode = "AUTO"
        else: 
            mode = "ANY" if gemini_declarations else "AUTO"
    else:
        logger.warning(f"RID-{request_id}: Google tool_choice: Invalid format {openai_tool_choice}. Defaulting to AUTO.")
        mode = "AUTO"

    function_calling_config: Dict[str, Any] = {"mode": mode}
    if mode == "ANY" and allowed_function_names:
        function_calling_config["allowed_function_names"] = allowed_function_names

    if gemini_declarations or mode == "NONE": 
        return {"function_calling_config": function_calling_config}
    elif mode == "AUTO" and not gemini_declarations: 
        return None
    
    return {"function_calling_config": function_calling_config}


def _convert_api_messages_to_gemini_contents(messages: List[ApiMessage], request_id: str) -> List[Dict[str, Any]]:
    gemini_contents = []
    for msg in messages:
        if msg.role == "user":
            gemini_contents.append({"role": "user", "parts": [{"text": msg.content or ""}]})
        elif msg.role == "assistant": 
            parts = []
            if msg.content is not None: 
                parts.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.type == "function" and tc.function.name and tc.function.arguments is not None:
                        try:
                            args_obj = orjson.loads(tc.function.arguments)
                        except orjson.JSONDecodeError:
                            args_obj = {"error": "Invalid JSON arguments from assistant", "raw_args": tc.function.arguments}
                            logger.warning(f"RID-{request_id}: Invalid JSON in tool_call args for func '{tc.function.name}' from assistant.")
                        parts.append({"functionCall": {"name": tc.function.name, "args": args_obj}})
            if parts: 
                gemini_contents.append({"role": "model", "parts": parts})
        elif msg.role == "tool": 
            if msg.name and msg.tool_call_id and msg.content is not None:
                try:
                    response_obj = orjson.loads(msg.content)
                except orjson.JSONDecodeError:
                    response_obj = {"raw_response": msg.content, "detail": "Content not valid JSON."}
                    logger.warning(f"RID-{request_id}: Tool response for '{msg.name}' not valid JSON.")

                gemini_contents.append({
                    "role": "function", 
                    "parts": [{"functionResponse": {"name": msg.name, "response": response_obj}}]
                })
    return gemini_contents


def prepare_openai_request(rd: ChatRequest, msgs_for_openai: List[ApiMessage], request_id: str) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    base = rd.api_address.strip() if rd.api_address else DEFAULT_OPENAI_API_BASE_URL
    url = f"{base.rstrip('/')}{OPENAI_COMPATIBLE_PATH}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {rd.api_key}"}
    
    openai_payload_msgs = []
    system_message_found_and_updated = False
    for m_obj in msgs_for_openai:
        msg_dict = {"role": m_obj.role}
        if m_obj.content is not None:
             msg_dict["content"] = m_obj.content
        elif m_obj.role == "system": 
            msg_dict["content"] = "" 

        if m_obj.tool_calls: msg_dict["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in m_obj.tool_calls]
        if m_obj.name: msg_dict["name"] = m_obj.name
        if m_obj.tool_call_id: msg_dict["tool_call_id"] = m_obj.tool_call_id
        
        if msg_dict.get("content") is None and not msg_dict.get("tool_calls") and msg_dict.get("role") != "system":
            continue

        if msg_dict.get("role") == "system":
            original_content = msg_dict.get("content", "")
            final_system_content = original_content
            if KATEX_FORMATTING_INSTRUCTION not in final_system_content:
                final_system_content = (final_system_content + "\n\n" + KATEX_FORMATTING_INSTRUCTION).strip()
            msg_dict["content"] = final_system_content
            system_message_found_and_updated = True
        openai_payload_msgs.append(msg_dict)

    if not system_message_found_and_updated:
        openai_payload_msgs.insert(0, {"role": "system", "content": KATEX_FORMATTING_INSTRUCTION})
        logger.info(f"RID-{request_id}: OpenAI Req: No system message, prepended KaTeX instruction.")

    payload: Dict[str, Any] = {"model": rd.model, "messages": openai_payload_msgs, "stream": True}
    if rd.temperature is not None: payload["temperature"] = rd.temperature
    if rd.top_p is not None: payload["top_p"] = rd.top_p
    if rd.max_tokens is not None: payload["max_tokens"] = rd.max_tokens
    if rd.tools: payload["tools"] = rd.tools
    if rd.tool_choice: payload["tool_choice"] = rd.tool_choice
    if rd.custom_model_parameters:
        logger.info(f"RID-{request_id}: OpenAI Req: Applying custom_model_parameters: {list(rd.custom_model_parameters.keys())}")
        for key, value in rd.custom_model_parameters.items():
            if key not in payload: payload[key] = value 
    if rd.custom_extra_body:
        logger.info(f"RID-{request_id}: OpenAI Req: Applying custom_extra_body: {list(rd.custom_extra_body.keys())}")
        payload.setdefault("extra_body", {}).update(rd.custom_extra_body)
    
    logger.debug(f"RID-{request_id}: OpenAI Request Payload (first 500 of messages): {str(payload.get('messages',[]))[:500]}")
    return url, headers, payload

def prepare_google_request_payload_structure(
    rd: ChatRequest, 
    api_messages: List[ApiMessage],
    request_id: str
) -> Tuple[Dict[str, Any], bool]: 
    
    generation_config_updates: Dict[str, Any] = {}
    is_native_gemini_thinking_active = False
    system_instruction_parts = []
    user_facing_messages: List[ApiMessage] = []
    has_client_system_message = False

    for m_obj in api_messages:
        if m_obj.role == "system" and m_obj.content: 
            has_client_system_message = True
            system_content_with_katex = f"{m_obj.content}\n\n{KATEX_FORMATTING_INSTRUCTION}"
            system_instruction_parts.append(system_content_with_katex)
        else:
            user_facing_messages.append(m_obj)

    if not has_client_system_message and ("gemini" in rd.model.lower()): 
        system_instruction_parts.append(KATEX_FORMATTING_INSTRUCTION)
    
    final_system_instruction_content = "\n\n".join(system_instruction_parts) if system_instruction_parts else None
    
    if is_gemini_2_5_model(rd.model):
        logger.info(f"RID-{request_id}: Model '{rd.model}' is Gemini 2.5 series. Enabling native thinking automatically for Google payload structure.")
        is_native_gemini_thinking_active = True
        thinking_config: Dict[str, Any] = {"includeThoughts": True}
        if rd.custom_model_parameters and "thinkingBudget" in rd.custom_model_parameters:
            try:
                budget = int(rd.custom_model_parameters["thinkingBudget"])
                if 0 <= budget <= 24576: 
                    thinking_config["thinkingBudget"] = budget
                    logger.info(f"RID-{request_id}: Using thinkingBudget: {budget}.")
            except (ValueError, TypeError):
                logger.warning(f"RID-{request_id}: thinkingBudget in custom_model_parameters is not a valid integer.")
        generation_config_updates["thinkingConfig"] = thinking_config
    
    gemini_api_contents = _convert_api_messages_to_gemini_contents(user_facing_messages, request_id)
    payload: Dict[str, Any] = {"contents": gemini_api_contents}

    if final_system_instruction_content:
        payload["systemInstruction"] = {"parts": [{"text": final_system_instruction_content}]}

    if rd.tools:
        gemini_declarations = _convert_openai_tools_to_gemini_declarations(rd.tools, request_id)
        if gemini_declarations:
            payload["tools"] = [{"functionDeclarations": gemini_declarations}]
            if rd.tool_choice: 
                tool_config_converted = _convert_openai_tool_choice_to_gemini_tool_config(rd.tool_choice, gemini_declarations, request_id)
                if tool_config_converted:
                    generation_config_updates.setdefault("toolConfig", {}).update(tool_config_converted) 

    if rd.temperature is not None: generation_config_updates["temperature"] = rd.temperature
    if rd.top_p is not None: generation_config_updates["topP"] = rd.top_p
    if rd.max_tokens is not None: generation_config_updates["maxOutputTokens"] = rd.max_tokens
    
    if generation_config_updates: 
        payload["generationConfig"] = generation_config_updates
        
    logger.debug(f"RID-{request_id}: Google Request Payload (first 500 of contents): {str(payload.get('contents',[]))[:500]}")
    return payload, is_native_gemini_thinking_active