# -*- coding: utf-8 -*-
"""
Gemini REST API request builder (thin, focused).

- Converts PartsApiMessagePy to REST API "contents".
- Injects render-safe V3 system prompt (English prompt body) when missing.
- Builds REST payload including tools and thinking config when applicable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ....core.config import GOOGLE_API_BASE_URL, CLOUDFLARE_BYPASS_STRATEGY
from ....models.api_models import (
    ChatRequestModel,
    SimpleTextApiMessagePy,
    PartsApiMessagePy,
    PyTextContentPart,
    PyInlineDataContentPart,
    PyFileUriContentPart,
    PyInputAudioContentPart,
)
from ....utils.helpers import is_gemini_2_5_model
from ..prompt_composer import (
    compose_system_prompt,
    detect_user_language_from_text,
    extract_user_texts_from_parts_messages,
)

logger = logging.getLogger("EzTalkProxy.Services.Requests.GeminiBuilder")


def should_enable_code_execution(chat_input: ChatRequestModel, request_id: str) -> bool:
    """
    判断是否应启用代码执行工具
    - enable_code_execution=True: 强制开启
    - enable_code_execution=False: 强制关闭
    - enable_code_execution=None: 智能判断（检测用户意图，所有 Gemini 系列模型均支持）
    """
    log_prefix = f"RID-{request_id}"
    
    # 显式控制
    if chat_input.enable_code_execution is True:
        logger.info(f"{log_prefix}: Code execution explicitly enabled")
        return True
    if chat_input.enable_code_execution is False:
        logger.info(f"{log_prefix}: Code execution explicitly disabled")
        return False
    
    # Auto模式：基于用户意图（所有 Gemini 系列模型均支持）
    model_lower = chat_input.model.lower()
    
    # 检查是否为 Gemini 系列模型
    if "gemini" not in model_lower:
        return False
    
    # 检测用户意图关键词
    user_texts = extract_user_texts_from_parts_messages(chat_input.messages)
    intent_keywords = [
        "计算", "求解", "运行代码", "执行代码", "画图", "绘制", "plot",
        "matplotlib", "数据分析", "统计", "csv", "pandas", "numpy",
        "calculate", "compute", "run code", "execute", "draw", "chart",
        "可视化", "visualization", "seaborn", "scipy"
    ]
    
    user_text_lower = user_texts.lower()
    if any(keyword in user_text_lower for keyword in intent_keywords):
        logger.info(f"{log_prefix}: Auto-enabled code execution for Gemini model based on user intent")
        return True
    
    return False


def convert_parts_messages_to_rest_api_contents(
    messages: List[PartsApiMessagePy],
    request_id: str
) -> List[Dict[str, Any]]:
    """
    convert_parts_messages_to_rest_api_contents(messages, request_id) -> List[dict]
    Convert internal PartsApiMessagePy messages to Gemini REST API 'contents'.
    """
    log_prefix = f"RID-{request_id}"
    rest_api_contents: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, PartsApiMessagePy):
            logger.warning(f"{log_prefix}: Expected PartsApiMessagePy at index {i}, got {type(msg)}. Skipping.")
            continue

        rest_parts: List[Dict[str, Any]] = []

        for actual_part in msg.parts:
            try:
                if isinstance(actual_part, PyTextContentPart):
                    rest_parts.append({"text": actual_part.text})
                elif isinstance(actual_part, PyInlineDataContentPart):
                    rest_parts.append({
                        "inlineData": {
                            "mimeType": actual_part.mime_type,
                            "data": actual_part.base64_data
                        }
                    })
                elif isinstance(actual_part, PyInputAudioContentPart):
                    mime_type = f"audio/{actual_part.format}"
                    rest_parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": actual_part.data
                        }
                    })
                elif isinstance(actual_part, PyFileUriContentPart):
                    if actual_part.uri.startswith("gs://"):
                        rest_parts.append({
                            "fileData": {
                                "mimeType": actual_part.mime_type,
                                "fileUri": actual_part.uri
                            }
                        })
                    else:
                        logger.warning(f"{log_prefix}: HTTP/S URI '{actual_part.uri}' for FileUriPart. REST API support varies. Skipping for now.")
                else:
                    logger.warning(f"{log_prefix}: Unknown actual part type during conversion: {type(actual_part)}. "
                                   f"Content: {str(actual_part)[:100]}. Skipping part.")
            except Exception as e_part:
                logger.error(f"{log_prefix}: Error processing message part for REST API: {actual_part}, Error: {e_part}", exc_info=True)

        if rest_parts:
            role_for_api = msg.role
            if msg.role == "assistant":
                role_for_api = "model"
            elif msg.role == "tool":
                role_for_api = "function"

            if role_for_api not in ["user", "model", "function"]:
                logger.warning(f"{log_prefix}: Mapping role '{msg.role}' to 'user' for Gemini REST API contents "
                               f"(current role_for_api: {role_for_api}).")
                role_for_api = "user"

            content_to_add = {"role": role_for_api, "parts": rest_parts}
            rest_api_contents.append(content_to_add)
        else:
            logger.warning(f"{log_prefix}: Message from role {msg.role} at index {i} resulted in no valid parts for REST API. Skipping.")

    return rest_api_contents


def add_system_prompt_to_gemini_messages(messages: List[PartsApiMessagePy], request_id: str) -> List[PartsApiMessagePy]:
    """
    add_system_prompt_to_gemini_messages(messages, request_id) -> List[PartsApiMessagePy]
    Inject the unified render-safe V3 system prompt (English body) as a 'system' PartsApiMessagePy
    when no system message exists.
    """
    log_prefix = f"RID-{request_id}"

    has_system_message = any((getattr(m, "role", "") or "").lower() == "system" for m in messages)

    if not has_system_message:
        user_text = extract_user_texts_from_parts_messages(messages)
        user_lang = detect_user_language_from_text(user_text)
        system_text = compose_system_prompt(False, user_lang)
        system_text_part = PyTextContentPart(type="text_content", text=system_text)
        system_message = PartsApiMessagePy(
            role="system",
            message_type="parts_message",
            parts=[system_text_part]
        )
        messages.insert(0, system_message)
        logger.info(f"{log_prefix}: Injected V3 system prompt for Gemini (lang={user_lang})")
    else:
        logger.info(f"{log_prefix}: System message exists for Gemini, skip injection")

    return messages


def prepare_gemini_rest_api_request(
    chat_input: ChatRequestModel,
    request_id: str,
    system_prompt: Optional[str] = None
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """
    Build Gemini REST API request:
    - Target URL (user-provided or Google official), with streamGenerateContent and SSE params
    - Headers with x-goog-api-key
    - 'contents' from PartsApiMessagePy (auto-inject V3 render-safe system prompt in parts format)
    - generationConfig and tools (googleSearch, functions)
    - Optional systemInstruction override when provided
    """
    log_prefix = f"RID-{request_id}"
    logger.info(f"{log_prefix}: Preparing Gemini REST API request for model {chat_input.model}.")

    model_name = chat_input.model

    # Require user-provided API key
    if not chat_input.api_key:
        raise ValueError("No user-provided API key for Gemini")

    base_api_url = GOOGLE_API_BASE_URL.rstrip('/')

    # Construct target URL
    if chat_input.api_address:
        if "/v1beta/models/" in chat_input.api_address:
            base_url = chat_input.api_address.rstrip('/')
            if ":generateContent" in base_url:
                target_url = base_url.replace(":generateContent", ":streamGenerateContent")
            else:
                target_url = base_url
            target_url = f"{target_url}?key={chat_input.api_key}&alt=sse"
            base_api_url = base_url.split('/v1beta/models/')[0]
        else:
            base_api_url = chat_input.api_address.rstrip('/')
            target_url = f"{base_api_url}/v1beta/models/{model_name}:streamGenerateContent?key={chat_input.api_key}&alt=sse"
    else:
        target_url = f"{base_api_url}/v1beta/models/{model_name}:streamGenerateContent?key={chat_input.api_key}&alt=sse"

    logger.info(f"{log_prefix}: Using user-provided API key for Gemini request to {base_api_url}")

    # 基础请求头
    # 注意：不设置 Accept-Encoding，让 httpx 自动处理压缩
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": chat_input.api_key,
    }
    
    # 根据配置策略添加浏览器特征头
    is_third_party_proxy = chat_input.api_address and "googleapis.com" not in chat_input.api_address.lower()
    
    if CLOUDFLARE_BYPASS_STRATEGY == "none":
        # 不添加任何额外头
        logger.info(f"{log_prefix}: Using minimal headers (strategy: none)")
    elif CLOUDFLARE_BYPASS_STRATEGY == "minimal":
        # 只添加基本的 User-Agent
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        headers["Accept"] = "application/json,*/*"
        logger.info(f"{log_prefix}: Using minimal browser headers (strategy: minimal)")
    else:  # "full" or default
        # 添加完整的浏览器特征头
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        headers["Accept"] = "text/event-stream,application/json,*/*"
        headers["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.8"
        headers["Cache-Control"] = "no-cache"
        headers["Pragma"] = "no-cache"
        
        # 如果是第三方代理，添加更多浏览器特征头
        if is_third_party_proxy:
            from urllib.parse import urlparse
            parsed = urlparse(chat_input.api_address)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            
            # 添加 Origin 和 Referer
            headers["Origin"] = origin
            headers["Referer"] = f"{origin}/"
            
            # 添加浏览器安全头（对代理服务器更宽松）
            headers["Sec-Ch-Ua"] = '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"'
            headers["Sec-Ch-Ua-Mobile"] = "?0"
            headers["Sec-Ch-Ua-Platform"] = '"Windows"'
            headers["Sec-Fetch-Dest"] = "empty"
            headers["Sec-Fetch-Mode"] = "cors"
            headers["Sec-Fetch-Site"] = "same-site"
            
            logger.info(f"{log_prefix}: Using full browser headers for proxy: {origin} (strategy: full)")
    json_payload: Dict[str, Any] = {}

    # Normalize messages to PartsApiMessagePy
    messages_to_convert_or_use: List[PartsApiMessagePy] = []
    for msg_abstract in chat_input.messages:
        if isinstance(msg_abstract, PartsApiMessagePy):
            messages_to_convert_or_use.append(msg_abstract)
        elif isinstance(msg_abstract, SimpleTextApiMessagePy):
            text_part = PyTextContentPart(type="text_content", text=msg_abstract.content or "")
            parts_message_equivalent = PartsApiMessagePy(
                role=msg_abstract.role,
                message_type="parts_message",
                parts=[text_part],
                name=msg_abstract.name,
                tool_calls=msg_abstract.tool_calls,
                tool_call_id=msg_abstract.tool_call_id
            )
            messages_to_convert_or_use.append(parts_message_equivalent)
        else:
            logger.warning(f"{log_prefix}: Unknown message type {type(msg_abstract)} in chat_input.messages. Skipping.")
    
    # Inject V3 render-safe system prompt (parts) when no explicit system exists
    messages_to_convert_or_use = add_system_prompt_to_gemini_messages(messages_to_convert_or_use, request_id)
    
    # Extract any 'system' messages into systemInstruction and REMOVE them from contents
    extracted_system_texts: List[str] = []
    remaining_messages: List[PartsApiMessagePy] = []
    for m in messages_to_convert_or_use:
        role_lower = (getattr(m, "role", "") or "").lower()
        if role_lower == "system":
            # collect text parts only
            for p in getattr(m, "parts", []) or []:
                try:
                    t = getattr(p, "text", None)
                    if t:
                        extracted_system_texts.append(str(t))
                except Exception:
                    continue
        else:
            remaining_messages.append(m)
    
    if extracted_system_texts and not system_prompt:
        # Prefer explicitly extracted system messages as systemInstruction
        sys_text = "\n\n".join([s for s in extracted_system_texts if s.strip()])
        if sys_text.strip():
            json_payload["systemInstruction"] = {"parts": [{"text": sys_text}]}
            logger.info(f"{log_prefix}: Moved {len(extracted_system_texts)} system message(s) into systemInstruction.")
    
    # Contents
    if not remaining_messages:
        logger.error(f"{log_prefix}: No processable non-system messages found for Gemini REST request.")
        json_payload["contents"] = []
    else:
        json_payload["contents"] = convert_parts_messages_to_rest_api_contents(remaining_messages, request_id)

    # Optional system instruction override (takes precedence over extracted)
    if system_prompt:
        json_payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    # Generation config
    generation_config_rest: Dict[str, Any] = {}
    if chat_input.generation_config:
        gc_in = chat_input.generation_config
        if gc_in.temperature is not None:
            generation_config_rest["temperature"] = gc_in.temperature
        if gc_in.top_p is not None:
            generation_config_rest["topP"] = gc_in.top_p
        if gc_in.max_output_tokens is not None:
            generation_config_rest["maxOutputTokens"] = gc_in.max_output_tokens
        if gc_in.thinking_config:
            tc_in = gc_in.thinking_config
            thinking_config_for_gen_config: Dict[str, Any] = {}
            if tc_in.include_thoughts is not None:
                thinking_config_for_gen_config["includeThoughts"] = tc_in.include_thoughts
            # 所有 Gemini 模型均支持 thinking_budget（移除版本限制）
            if tc_in.thinking_budget is not None and "gemini" in model_name.lower():
                thinking_config_for_gen_config["thinkingBudget"] = tc_in.thinking_budget
            if thinking_config_for_gen_config:
                generation_config_rest["thinkingConfig"] = thinking_config_for_gen_config

    if "temperature" not in generation_config_rest and chat_input.temperature is not None:
        generation_config_rest["temperature"] = chat_input.temperature
    if "topP" not in generation_config_rest and chat_input.top_p is not None:
        generation_config_rest["topP"] = chat_input.top_p
    if "maxOutputTokens" not in generation_config_rest and chat_input.max_tokens is not None:
        generation_config_rest["maxOutputTokens"] = chat_input.max_tokens

    if generation_config_rest:
        json_payload["generationConfig"] = generation_config_rest

    # Tools
    gemini_tools_payload: List[Dict[str, Any]] = []
    if chat_input.use_web_search:
        gemini_tools_payload.append({"googleSearch": {}})
        logger.info(f"{log_prefix}: Enabled Google Search tool for Gemini.")
    
    # 代码执行工具
    if should_enable_code_execution(chat_input, request_id):
        gemini_tools_payload.append({"codeExecution": {}})
        logger.info(f"{log_prefix}: Enabled Code Execution tool for Gemini.")

    if chat_input.tools:
        converted_declarations: List[Dict[str, Any]] = []
        for tool_entry in chat_input.tools:
            if tool_entry.get("type") == "function" and "function" in tool_entry:
                func_data = tool_entry["function"]
                declaration = {
                    "name": func_data.get("name"),
                    "description": func_data.get("description"),
                    "parameters": func_data.get("parameters"),
                }
                declaration = {k: v for k, v in declaration.items() if v is not None}
                if "name" in declaration and "description" in declaration:
                    converted_declarations.append(declaration)

        if converted_declarations:
            gemini_tools_payload.append({"functionDeclarations": converted_declarations})

    if gemini_tools_payload:
        json_payload["tools"] = gemini_tools_payload
        if chat_input.tool_choice:
            tool_config_payload: Dict[str, Any] = {}
            if isinstance(chat_input.tool_choice, str):
                choice_str = chat_input.tool_choice.upper()
                if choice_str in ["AUTO", "ANY", "NONE"]:
                    tool_config_payload = {"mode": choice_str}
                elif choice_str == "REQUIRED":
                    tool_config_payload = {"mode": "ANY"}
            elif isinstance(chat_input.tool_choice, dict) and chat_input.tool_choice.get("type") == "function":
                func_choice = chat_input.tool_choice.get("function", {})
                func_name = func_choice.get("name")
                if func_name:
                    tool_config_payload = {"mode": "ANY", "allowedFunctionNames": [func_name]}

            if tool_config_payload:
                if "generationConfig" not in json_payload:
                    json_payload["generationConfig"] = {}
                json_payload["generationConfig"]["toolConfig"] = {"functionCallingConfig": tool_config_payload}

    logger.info(f"{log_prefix}: Prepared Gemini REST API request. URL: {target_url.split('?key=')[0]}... "
                f"Payload keys: {list(json_payload.keys())}")
    if "generationConfig" in json_payload:
        logger.info(f"{log_prefix}: generationConfig in REST payload: {json_payload['generationConfig']}")

    return target_url, headers, json_payload