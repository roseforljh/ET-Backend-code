"""
Requests building package.

Exports facade entries that mirror legacy services.request_builder functions.
See [facade.py](backdAiTalk/eztalk_proxy/services/requests/facade.py) for delegation during migration.
"""

from .facade import (
    prepare_openai_request,
    prepare_gemini_rest_api_request,
    add_system_prompt_if_needed,
    convert_parts_messages_to_rest_api_contents,
    add_system_prompt_to_gemini_messages,
)

__all__ = [
    "prepare_openai_request",
    "prepare_gemini_rest_api_request",
    "add_system_prompt_if_needed",
    "convert_parts_messages_to_rest_api_contents",
    "add_system_prompt_to_gemini_messages",
]