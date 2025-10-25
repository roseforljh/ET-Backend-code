"""
Requests building facade (post-refactor).

This module exposes stable functions that mirror the legacy services.request_builder
API. It now delegates directly to the new, smaller, focused builders in the
`builders` sub-package.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

# Direct imports from the new, focused builders
from .builders import (
    prepare_openai_request,
    add_system_prompt_if_needed,
    prepare_gemini_rest_api_request,
    convert_parts_messages_to_rest_api_contents,
    add_system_prompt_to_gemini_messages,
)

logger = logging.getLogger("EzTalkProxy.Requests.Facade")

# The facade functions now directly call the imported builder functions.
# This maintains the external API while pointing to the new, refactored logic.
# No legacy fallback or NotImplementedError is needed anymore.

__all__ = [
    "prepare_openai_request",
    "add_system_prompt_if_needed",
    "prepare_gemini_rest_api_request",
    "convert_parts_messages_to_rest_api_contents",
    "add_system_prompt_to_gemini_messages",
]