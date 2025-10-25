"""
Converters between internal message models and provider REST API payloads.

During migration, these helpers delegate to legacy implementations to keep
behavior identical while paving the path for future refactors.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Delegate to legacy converter to preserve behavior
try:
    from ..request_builder import (
        convert_parts_messages_to_rest_api_contents as _legacy_convert_parts_messages_to_rest_api_contents,
    )
except Exception:
    _legacy_convert_parts_messages_to_rest_api_contents = None  # type: ignore[assignment]


def convert_parts_messages_to_rest_api_contents(messages: List[Any], request_id: str) -> List[Dict[str, Any]]:
    """
    Convert PartsApiMessagePy messages into REST API 'contents' as expected by Gemini REST.

    Delegates to legacy implementation during migration.
    """
    if _legacy_convert_parts_messages_to_rest_api_contents is None:
        return []
    return _legacy_convert_parts_messages_to_rest_api_contents(messages, request_id)