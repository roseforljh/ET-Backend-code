"""
Flush utilities for streaming - simplified to always flush immediately.
"""

from __future__ import annotations


def inside_code_fence(text: str) -> bool:
    """简单判断当前是否处于未闭合的代码围栏中（``` 计数为奇数）"""
    if not text:
        return False
    return text.count("```") % 2 == 1


class MarkdownBlockDetector:
    """简化的检测器 - 统一策略：立即刷新所有内容"""

    @staticmethod
    def is_complete_code_block(text: str) -> bool:
        """检测代码块是否完整"""
        if not text.strip():
            return False
        lines = text.split("\n")
        fence_count = 0
        for line in lines:
            if line.strip().startswith("```"):
                fence_count += 1
        return fence_count >= 2 and fence_count % 2 == 0

    @staticmethod
    def is_safe_flush_point(accumulated_content: str, new_chunk: str, mode: str = None) -> bool:
        """
        统一策略：始终返回 True，立即刷新所有内容
        保留函数签名以兼容现有代码
        """
        return True

    @staticmethod
    def detect_block_type(text: str) -> str:
        """检测块类型（简化版）"""
        if not text or not text.strip():
            return "text"

        text_lower = text.lower().strip()

        if text_lower.startswith("```"):
            return "code_block"
        elif text_lower.startswith("#"):
            return "heading"
        else:
            return "text"


__all__ = ["inside_code_fence", "MarkdownBlockDetector"]
