"""
智能句子分割器

用于语音模式中的 LLM 输出分割，以提前触发 TTS，减少首次语音输出延迟。

分割策略优先级：
1. 强制分割点：句末标点（。！？.!?）、换行
2. 推荐分割点：逗号、分号等 + 满足最小长度
3. 紧急分割点：超过最大长度时强制在任意标点处分割
4. 兜底分割：超过绝对最大长度时在空格/字符边界分割
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger("EzTalkProxy.Services.SmartSplitter")


@dataclass
class SplitResult:
    """分割结果"""
    segments: List[str]    # 可以发送 TTS 的片段
    remainder: str         # 剩余未完成的 buffer


class SmartSentenceSplitter:
    """
    智能句子分割器
    
    特点：
    - 尽早触发 TTS，减少延迟
    - 保持语义完整性
    - 自然的语音节奏
    - 处理特殊情况（数字、引用等）
    """
    
    def __init__(
        self,
        min_length: int = 8,
        preferred_length: int = 20,
        max_length: int = 50,
        absolute_max: int = 80
    ):
        """
        初始化分割器
        
        Args:
            min_length: 最小分割长度（避免太碎）
            preferred_length: 理想分割长度
            max_length: 最大等待长度（超过后在任意标点分割）
            absolute_max: 绝对最大长度（强制分割）
        """
        self.min_length = min_length
        self.preferred_length = preferred_length
        self.max_length = max_length
        self.absolute_max = absolute_max
        
        # 分割点正则
        self.strong_endings = re.compile(r'[。！？.!?\n]')
        self.weak_endings = re.compile(r'[，,；;：:、]')
        self.any_punctuation = re.compile(r'[，,；;：:、。！？.!?\n]')
        
        # 保护模式：这些模式匹配时，尾部的标点不应作为分割点
        self.protected_tail_patterns = [
            re.compile(r'\d+[.,]$'),           # 小数未完成：3.
            re.compile(r'[\d]+[:：]$'),         # 时间未完成：12:
            re.compile(r'[《「『【][^》」』】]*$'),  # 未闭合的书名号
            re.compile(r'"[^"]*$'),              # 未闭合的中文引号
            re.compile(r'"[^"]*$'),              # 未闭合的英文引号
            re.compile(r"'[^']*$"),              # 未闭合的单引号
        ]
        
        # 不应断开的短语前缀（尾部匹配）
        self.no_break_prefixes = [
            "请稍", "稍等", "让我", "我来",
            "比如", "例如", "也就", "就是",
            "首先", "其次", "最后", "然后",
            "一方", "另一", "此外", "另外",
            "因为", "所以", "但是", "而且",
            "如果", "那么", "否则",
        ]
        
        logger.debug(f"SmartSentenceSplitter initialized: min={min_length}, preferred={preferred_length}, max={max_length}")
    
    def split(self, buffer: str) -> SplitResult:
        """
        分割 buffer，返回可发送的片段和剩余内容
        
        Args:
            buffer: 待分割的文本
            
        Returns:
            SplitResult: 包含可发送片段列表和剩余内容
        """
        segments = []
        remaining = buffer
        
        while remaining:
            split_point = self._find_best_split_point(remaining)
            
            if split_point is None:
                # 无法分割，全部保留
                break
            
            if split_point <= 0:
                # 保护模式激活，等待更多内容
                break
            
            # 提取片段
            segment = remaining[:split_point].strip()
            remaining = remaining[split_point:].lstrip()
            
            if segment:
                segments.append(segment)
                logger.debug(f"Extracted segment ({len(segment)} chars): '{segment[:30]}...'")
        
        return SplitResult(segments=segments, remainder=remaining)
    
    def _find_best_split_point(self, text: str) -> Optional[int]:
        """
        寻找最佳分割点
        
        Returns:
            int: 分割位置（包含标点）
            None: 不分割，继续等待
            0: 在保护区内，继续等待
        """
        length = len(text)
        
        # 1. 检查是否处于保护模式
        if self._is_protected(text):
            logger.debug(f"Buffer is protected, waiting for more content")
            return 0
        
        # 2. 寻找强制分割点（句末标点）
        strong_match = None
        for m in self.strong_endings.finditer(text):
            pos = m.end()
            # 检查分割后的长度是否合理
            if pos >= self.min_length:
                # 检查该位置是否在保护区内
                if not self._is_position_protected(text, m.start()):
                    strong_match = pos
                    break  # 找到第一个有效的强分割点
        
        if strong_match:
            logger.debug(f"Found strong split point at {strong_match}")
            return strong_match
        
        # 3. 如果长度足够，寻找推荐分割点（逗号、分号等）
        if length >= self.min_length:
            # 优先在理想长度附近分割
            best_weak = None
            best_distance = float('inf')
            
            for m in self.weak_endings.finditer(text):
                pos = m.end()
                if pos >= self.min_length:
                    if not self._is_position_protected(text, m.start()):
                        distance = abs(pos - self.preferred_length)
                        if distance < best_distance:
                            best_distance = distance
                            best_weak = pos
            
            # 如果找到合适的弱分割点，且长度超过理想长度，执行分割
            if best_weak and length >= self.preferred_length:
                logger.debug(f"Found weak split point at {best_weak} (preferred={self.preferred_length})")
                return best_weak
        
        # 4. 超长保护：超过最大长度时，在任意标点处分割
        if length >= self.max_length:
            for m in self.any_punctuation.finditer(text):
                pos = m.end()
                if pos >= self.min_length:
                    logger.debug(f"Forced split at {pos} due to max_length={self.max_length}")
                    return pos
        
        # 5. 绝对最大保护：在空格或字符边界强制分割
        if length >= self.absolute_max:
            # 优先在空格处分割
            space_pos = text.rfind(' ', self.min_length, self.absolute_max)
            if space_pos > 0:
                logger.debug(f"Emergency split at space position {space_pos + 1}")
                return space_pos + 1
            # 否则在绝对最大位置硬切
            logger.debug(f"Hard cut at absolute_max={self.absolute_max}")
            return self.absolute_max
        
        # 不分割，继续等待
        return None
    
    def _is_protected(self, text: str) -> bool:
        """检查 buffer 尾部是否处于保护模式"""
        # 检查尾部模式
        for pattern in self.protected_tail_patterns:
            if pattern.search(text):
                return True
        
        # 检查尾部是否匹配不应断开的短语前缀
        for prefix in self.no_break_prefixes:
            if text.endswith(prefix):
                return True
        
        return False
    
    def _is_position_protected(self, text: str, pos: int) -> bool:
        """检查特定位置的标点是否在保护区内"""
        if pos <= 0 or pos >= len(text):
            return False
            
        before = text[pos - 1] if pos > 0 else ''
        after = text[pos + 1] if pos + 1 < len(text) else ''
        
        # 小数点保护：前后都是数字
        if before.isdigit() and after.isdigit():
            return True
        
        # 千分位逗号保护
        char = text[pos]
        if char == ',' and before.isdigit() and after.isdigit():
            return True
        
        return False


# 全局默认实例
default_splitter = SmartSentenceSplitter()


def split_for_tts(buffer: str) -> SplitResult:
    """
    便捷函数：使用默认配置分割文本
    
    Args:
        buffer: 待分割的文本
        
    Returns:
        SplitResult: 分割结果
    """
    return default_splitter.split(buffer)