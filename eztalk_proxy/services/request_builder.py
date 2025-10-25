# -*- coding: utf-8 -*-
"""
Request builder legacy facade (thin proxy).

This module previously contained large request building implementations.
It has been refactored to delegate all logic to smaller, focused modules
under the `requests/builders/` package.

This file is kept to provide a stable import path for external callers,
preserving the API contract while allowing internal logic to be modular.
All functions defined here now delegate to their counterparts in the new
`builders` package.
"""

# === Thin delegation to split modules (post-refactor) ===
# NOTE: These overrides intentionally appear at file end to take precedence
# over earlier same-named definitions in this legacy module. This keeps
# external imports stable while switching behavior to new smaller modules.

try:
    from .requests.builders.openai_builder import (
        prepare_openai_request as _openai_prepare_openai_request,
        add_system_prompt_if_needed as _openai_add_system_prompt_if_needed,
    )
    from .requests.builders.gemini_builder import (
        prepare_gemini_rest_api_request as _gemini_prepare_gemini_rest_api_request,
        convert_parts_messages_to_rest_api_contents as _gemini_convert_parts_messages_to_rest_api_contents,
        add_system_prompt_to_gemini_messages as _gemini_add_system_prompt_to_gemini_messages,
    )
    _SPLIT_BUILDERS_AVAILABLE = True
except Exception as _e:
    _SPLIT_BUILDERS_AVAILABLE = False

if _SPLIT_BUILDERS_AVAILABLE:
    # OpenAI-format: system prompt injection (English body, render-safe V3)
    def add_system_prompt_if_needed(messages, request_id):
        return _openai_add_system_prompt_if_needed(messages, request_id)

    # OpenAI-format: request builder
    def prepare_openai_request(request_data, processed_messages, request_id, system_prompt=None):
        return _openai_prepare_openai_request(request_data, processed_messages, request_id, system_prompt)

    # Gemini REST: message conversion
    def convert_parts_messages_to_rest_api_contents(messages, request_id):
        return _gemini_convert_parts_messages_to_rest_api_contents(messages, request_id)

    # Gemini REST: system prompt injection (English body, render-safe V3)
    def add_system_prompt_to_gemini_messages(messages, request_id):
        return _gemini_add_system_prompt_to_gemini_messages(messages, request_id)

    # Gemini REST: request builder
    def prepare_gemini_rest_api_request(chat_input, request_id, system_prompt=None):
        return _gemini_prepare_gemini_rest_api_request(chat_input, request_id, system_prompt)