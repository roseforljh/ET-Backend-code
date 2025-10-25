# -*- coding: utf-8 -*-
"""
OpenAI-format request builder (thin, focused).

- Injects render-safe V3 system prompt (English prompt body) when missing.
- Builds OpenAI-compatible payload with tool support and optional Gemini-in-OpenAI-format extras.
- Enables Gemini native google_search tool when use_web_search is enabled.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from ....utils.helpers import is_gemini_2_5_model  # not used here but kept for parity
from ....core.config import DEFAULT_OPENAI_API_BASE_URL
from ....models.api_models import ChatRequestModel
from ..prompt_composer import (
    compose_system_prompt,
    detect_user_language_from_text,
    extract_user_texts_from_openai_messages,
)

logger = logging.getLogger("EzTalkProxy.Services.Requests.OpenAIBuilder")


def is_gemini_model_in_openai_format(model_name: str) -> bool:
    """Detect if the target model is a Gemini model called via OpenAI-compatible schema."""
    if not model_name:
        return False
    return "gemini" in model_name.lower()


def add_system_prompt_if_needed(messages: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    """
    add_system_prompt_if_needed(messages: List[dict], request_id: str) -> List[dict]
    Inject the unified render-safe V3 system prompt (English body) based on user intent/language.
    Only inject when there is no existing system message.
    """
    log_prefix = f"RID-{request_id}"
    has_system_message = any((msg.get("role") or "").lower() == "system" for msg in messages)

    if not has_system_message:
        user_text = extract_user_texts_from_openai_messages(messages)
        user_lang = detect_user_language_from_text(user_text)
        system_text = compose_system_prompt(False, user_lang)
        system_message = {"role": "system", "content": system_text}
        messages.insert(0, system_message)
        logger.info(
            f"{log_prefix}: Injected V3 system prompt (lang={user_lang})"
        )
    else:
        logger.info(f"{log_prefix}: System message already exists, skip injection")

    return messages


def prepare_openai_request(
    request_data: ChatRequestModel,
    processed_messages: List[Dict[str, Any]],
    request_id: str,
    system_prompt: Optional[str] = None,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """
    Build OpenAI-compatible request:
    - Authorization headers (+ x-api-key, and x-goog-api-key for Gemini-in-OpenAI-format)
    - Streaming enabled
    - Tools & optional vendor-specific extras
    - V3 render-safe system prompt auto-injection (English prompt body)
    - Gemini native google_search tool when use_web_search is enabled
    """
    target_url = request_data.api_address or DEFAULT_OPENAI_API_BASE_URL

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {request_data.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "x-api-key": request_data.api_key,
    }
    
    is_gemini = is_gemini_model_in_openai_format(request_data.model)
    if is_gemini:
        headers["x-goog-api-key"] = request_data.api_key

    final_messages = add_system_prompt_if_needed(copy.deepcopy(processed_messages), request_id)
    if system_prompt:
        final_messages.insert(0, {"role": "system", "content": system_prompt})

    payload: Dict[str, Any] = {
        "model": request_data.model,
        "messages": final_messages,
        "stream": True,
    }

    gen_conf = request_data.generation_config
    if gen_conf:
        payload.update(
            {
                "temperature": gen_conf.temperature,
                "top_p": gen_conf.top_p,
                "max_tokens": gen_conf.max_output_tokens,
            }
        )

    payload.update(
        {
            "temperature": payload.get("temperature") or request_data.temperature,
            "top_p": payload.get("top_p") or request_data.top_p,
            "max_tokens": payload.get("max_tokens") or request_data.max_tokens,
            "tools": request_data.tools,
            "tool_choice": request_data.tool_choice,
        }
    )

    # Gemini-in-OpenAI-format: optional "thinking" extras for some aggregators
    if is_gemini:
        try:
            tc = getattr(gen_conf, "thinking_config", None) if gen_conf else None
            google_thinking_cfg: Dict[str, Any] = {}
            if tc is not None:
                if getattr(tc, "include_thoughts", None) is not None:
                    google_thinking_cfg["include_thoughts"] = tc.include_thoughts
                if getattr(tc, "thinking_budget", None) is not None:
                    google_thinking_cfg["thinking_budget"] = tc.thinking_budget
            if google_thinking_cfg:
                extra_body = payload.get("extra_body") or {}
                google_section = extra_body.get("google") or {}
                google_section["thinking_config"] = google_thinking_cfg
                extra_body["google"] = google_section
                payload["extra_body"] = extra_body
            if (not google_thinking_cfg.get("thinking_budget")) and (tc is not None and getattr(tc, "include_thoughts", False)):
                if "reasoning_effort" not in payload:
                    payload["reasoning_effort"] = "low"
        except Exception as _e:
            logger.warning(f"RID-{request_id}: Failed to attach Gemini thinking_config (OpenAI format): {_e}")
        
        # 🔥 NEW: Enable Gemini native google_search tool when use_web_search is enabled
        if request_data.use_web_search:
            extra_body = payload.get("extra_body") or {}
            google_section = extra_body.get("google") or {}
            google_tools = google_section.get("tools") or []
            
            # Add google_search tool if not already present
            if not any(tool.get("google_search") is not None for tool in google_tools):
                google_tools.append({"google_search": {}})
                google_section["tools"] = google_tools
                extra_body["google"] = google_section
                payload["extra_body"] = extra_body
                logger.info(f"RID-{request_id}: Enabled Gemini native google_search tool in OpenAI-compatible format")
        
        # 🔥 NEW: Enable Gemini native code_execution tool when enabled
        from ..builders.gemini_builder import should_enable_code_execution
        if should_enable_code_execution(request_data, request_id):
            extra_body = payload.get("extra_body") or {}
            google_section = extra_body.get("google") or {}
            google_tools = google_section.get("tools") or []
            
            # Add code_execution tool if not already present
            if not any(tool.get("code_execution") is not None for tool in google_tools):
                google_tools.append({"code_execution": {}})
                google_section["tools"] = google_tools
                extra_body["google"] = google_section
                payload["extra_body"] = extra_body
                logger.info(f"RID-{request_id}: Enabled Gemini native code_execution tool in OpenAI-compatible format")

    # Vendor toggles
    if "qwen" in request_data.model.lower() and isinstance(request_data.qwen_enable_search, bool):
        payload["enable_search"] = request_data.qwen_enable_search

    # Custom model params / extra body
    if request_data.custom_model_parameters:
        for key, value in request_data.custom_model_parameters.items():
            if key not in payload:
                payload[key] = value
    if request_data.custom_extra_body:
        payload.update(request_data.custom_extra_body)

    # Drop Nones
    payload = {k: v for k, v in payload.items() if v is not None}

    return target_url, headers, payload