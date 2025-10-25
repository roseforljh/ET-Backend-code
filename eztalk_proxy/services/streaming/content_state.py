"""
Content state utilities extracted from legacy stream_processor.

Provides:
- compute_content_hash(content: str) -> str
- is_duplicate_content(new_content: str, previous_content: str, threshold: float = 0.8) -> bool
"""

from __future__ import annotations

import hashlib
from typing import Optional


def compute_content_hash(content: str) -> str:
    """计算内容哈希用于去重（与旧实现保持一致）"""
    if not content:
        return ""
    return hashlib.md5(content.strip().encode("utf-8")).hexdigest()


def is_duplicate_content(new_content: Optional[str], previous_content: Optional[str], threshold: float = 0.8) -> bool:
    """
    检测是否为重复内容（从 services.stream_processor 迁移，保持语义一致）

    Args:
        new_content: 新内容
        previous_content: 之前的内容
        threshold: 相似度阈值(0-1)

    Returns:
        True 如果内容重复，否则 False
    """
    if not new_content or not previous_content:
        return False

    new_clean = new_content.strip()
    prev_clean = previous_content.strip()

    # 完全相同
    if new_clean == prev_clean:
        return True

    # 新内容完全包含在之前内容中（长度>10 时启用）
    if len(new_clean) > 10 and new_clean in prev_clean:
        return True

    # 检查是否是重复的段落（逐行对比）
    new_lines = [line.strip() for line in new_clean.split("\n") if line.strip()]
    prev_lines = [line.strip() for line in prev_clean.split("\n") if line.strip()]

    if len(new_lines) > 0:
        duplicate_lines = sum(1 for line in new_lines if line in prev_lines)
        if duplicate_lines / len(new_lines) > threshold:
            return True

    return False