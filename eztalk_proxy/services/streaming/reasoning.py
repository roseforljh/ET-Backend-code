"""
Reasoning helpers extracted from legacy stream_processor.

Provides:
- extract_think_tags(text) -> tuple[str, str]
- should_extract_think_tags_from_content(request_data, request_id) -> bool

These functions were originally defined in services.stream_processor and are
moved here to start decoupling the large monolith into focused modules.
"""

from __future__ import annotations

import re
import logging
from typing import Tuple, Any

logger = logging.getLogger("EzTalkProxy.Streaming.Reasoning")


def extract_think_tags(text: str) -> Tuple[str, str]:
    """
    从文本中提取<think>标签内容

    Returns:
        Tuple[思考内容, 剩余内容]
    """
    if not text or '<think>' not in text:
        return "", text

    # 匹配所有<think>...</think>标签（支持多个标签和换行）
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, text, re.DOTALL | re.IGNORECASE)

    if not matches:
        return "", text

    # 提取所有思考内容
    thinking_content = "\n\n".join(match.strip() for match in matches if match.strip())

    # 移除所有<think>标签及其内容，得到剩余内容
    remaining_content = re.sub(think_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    remaining_content = remaining_content.strip()

    return thinking_content, remaining_content


def should_extract_think_tags_from_content(request_data: Any, request_id: str) -> bool:
    """
    判断是否应该从content中提取<think>标签
    主要针对DeepSeek等在content中包含<think>标签的模型
    """
    log_prefix = f"RID-{request_id}"

    if not hasattr(request_data, 'model') or not request_data.model:
        return False

    model_lower = str(request_data.model).lower()

    # DeepSeek模型经常在content中使用<think>标签
    if "deepseek" in model_lower:
        logger.info(f"{log_prefix}: Enabling <think> tag extraction for DeepSeek model: {request_data.model}")
        return True

    # MKE提供商也可能使用这种格式
    if hasattr(request_data, 'provider') and request_data.provider and str(request_data.provider).lower() == "mke":
        logger.info(f"{log_prefix}: Enabling <think> tag extraction for MKE provider")
        return True

    # 其他可能使用<think>标签的模型可以在这里添加
    think_tag_models = ['qwen', 'claude', 'gpt', 'gemini']  # 一些可能使用思考标签的模型
    for model_keyword in think_tag_models:
        if model_keyword in model_lower:
            logger.info(f"{log_prefix}: Enabling <think> tag extraction for model containing '{model_keyword}': {request_data.model}")
            return True

    return False