"""
Delta utilities extracted from legacy stream_processor.

Provides:
- merge_incremental_text(accumulated: str, incoming: str) -> str
  合并增量文本，尽量避免重复：
  - 若 incoming 以 accumulated 开头：返回 incoming（供应商以“全量替换”方式推送）
  - 否则寻找 accumulated 的最长后缀 = incoming 的前缀的重叠部分，仅追加非重叠尾部
  - 若无重叠：直接累加（保底）
"""

from __future__ import annotations


def merge_incremental_text(accumulated: str, incoming: str) -> str:
    """
    合并增量文本，尽量避免重复。
    与旧实现保持一致的语义，便于平滑迁移。
    """
    if not accumulated:
        return incoming
    if not incoming:
        return accumulated

    # 情况1：上游每次传“至今为止的完整文本”
    if incoming.startswith(accumulated):
        return incoming

    # 情况2：寻找最大重叠（acc 后缀 == inc 前缀）
    max_overlap = 0
    max_check = min(len(accumulated), len(incoming))
    # 从较长的重叠长度开始尝试，尽快命中
    for l in range(max_check, 0, -1):
        if accumulated.endswith(incoming[:l]):
            max_overlap = l
            break

    if max_overlap > 0:
        return accumulated + incoming[max_overlap:]

    # 情况3：无明显重叠，保底拼接（仍然比简单相加可靠，因为已排除“全量替换”模式）
    return accumulated + incoming