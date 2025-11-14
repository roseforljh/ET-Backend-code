# -*- coding: utf-8 -*-
"""
Unified prompts module (consolidated).

Single source of truth for all prompt-related helpers.
Currently implemented as no-ops to avoid implicit system prompt injection.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Backward-compat constant (kept but empty by design)
RENDER_SAFE_V3_PROMPT_EN: str = ""


def compose_system_prompt(is_math: bool, user_language: str) -> str:
    """
    compose_system_prompt(is_math: bool, user_language: str) -> str
    NO-OP: returns empty string to avoid implicit system prompt injection.
    """
    return ""


def add_system_prompt_if_needed(messages: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    """
    Identity: return messages as-is (do not inject system prompt).
    """
    return messages


def add_system_prompt_to_gemini_messages(messages: List[Any], request_id: str) -> List[Any]:
    """
    Identity: return messages as-is (do not inject system prompt).
    """
    return messages


def detect_math_intent(text: str) -> bool:
    """
    Always returns False (disabled).
    """
    return False


def detect_user_language_from_text(text: str) -> str:
    """
    Lightweight language detection; returns a BCP-47-like tag.
    (Kept for compatibility; does not trigger any injection.)
    """
    if not text:
        return "en"
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:  # CJK Unified Ideographs
            return "zh-CN"
        if 0x3040 <= cp <= 0x309F or 0x30A0 <= 0x30FF:  # Hiragana/Katakana
            return "ja-JP"
        if 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F or 0xAC00 <= cp <= 0xD7AF:  # Hangul
            return "ko-KR"
        if 0x0400 <= cp <= 0x04FF:  # Cyrillic
            return "ru-RU"
        if 0x0600 <= cp <= 0x06FF:  # Arabic
            return "ar"
        if 0x0900 <= cp <= 0x097F:  # Devanagari
            return "hi-IN"
    return "en"


def extract_user_texts_from_openai_messages(messages: List[dict]) -> str:
    """
    Collects user text from OpenAI-style message arrays (including array content parts).
    Compatibility shim; trimmed to 4000 chars.
    """
    texts: List[str] = []
    for msg in messages or []:
        try:
            if (msg.get("role") or "").lower() == "user":
                c = msg.get("content")
                if isinstance(c, str):
                    texts.append(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            texts.append(part["text"])
        except Exception:
            continue
    return "\n".join(texts)[:4000]


def extract_user_texts_from_parts_messages(messages: List[object]) -> str:
    """
    Collects user text from PartsApiMessagePy-style messages (text parts only).
    Compatibility shim; trimmed to 4000 chars.
    """
    texts: List[str] = []
    for m in messages or []:
        try:
            if (getattr(m, "role", "") or "").lower() == "user":
                for p in getattr(m, "parts", []) or []:
                    if getattr(p, "type", None) in (None, "text_content") and hasattr(p, "text"):
                        texts.append(getattr(p, "text", "") or "")
        except Exception:
            continue
    return "\n".join(texts)[:4000]