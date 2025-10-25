# -*- coding: utf-8 -*-
"""
Request builders package.

Contains thin, focused builders for different provider protocols
and formats (OpenAI-compatible, Gemini REST, etc.).
"""
from .openai_builder import (
    prepare_openai_request,
    add_system_prompt_if_needed,
)
from .gemini_builder import (
    prepare_gemini_rest_api_request,
    convert_parts_messages_to_rest_api_contents,
    add_system_prompt_to_gemini_messages,
)

__all__ = [
    "prepare_openai_request",
    "add_system_prompt_if_needed",
    "prepare_gemini_rest_api_request",
    "convert_parts_messages_to_rest_api_contents",
    "add_system_prompt_to_gemini_messages",
]