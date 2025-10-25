"""
System prompt helpers (intent/language aware) for request building.

Simplified version without math intent detection.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Delegate to legacy helpers to keep behavior identical
try:
    from ..request_builder import (
        add_system_prompt_if_needed as _legacy_add_system_prompt_if_needed,
        add_system_prompt_to_gemini_messages as _legacy_add_system_prompt_to_gemini_messages,
        _detect_user_language_from_text as _legacy_detect_user_lang,    # type: ignore[attr-defined]
    )
except Exception as _e:  # pragma: no cover - defensive
    _legacy_add_system_prompt_if_needed = None  # type: ignore[assignment]
    _legacy_add_system_prompt_to_gemini_messages = None  # type: ignore[assignment]
    _legacy_detect_user_lang = None  # type: ignore[assignment]


def add_system_prompt_if_needed(messages: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    """
    OpenAI-format messages: inject unified system prompt when needed.
    """
    if _legacy_add_system_prompt_if_needed is None:
        return messages
    return _legacy_add_system_prompt_if_needed(messages, request_id)


def add_system_prompt_to_gemini_messages(messages: List[Any], request_id: str) -> List[Any]:
    """
    Gemini parts-format messages: inject unified system prompt when needed.
    """
    if _legacy_add_system_prompt_to_gemini_messages is None:
        return messages
    return _legacy_add_system_prompt_to_gemini_messages(messages, request_id)


def detect_math_intent(text: str) -> bool:
    """
    Simplified: always returns False (math intent detection disabled).
    """
    return False


def detect_user_language_from_text(text: str) -> str:
    """
    Lightweight proxy for user language detection.
    """
    if _legacy_detect_user_lang is None:
        return "en"
    return _legacy_detect_user_lang(text)
