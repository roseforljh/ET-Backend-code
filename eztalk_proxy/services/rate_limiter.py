"""
速率限制服务
用于限制特定模型的使用频率
"""
import time
import logging
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
import threading

from ..core.config import DEFAULT_TEXT_MODELS

logger = logging.getLogger("EzTalkProxy.RateLimiter")


def _normalize_provider(provider: Optional[str]) -> str:
    if not provider:
        return ""
    return (provider or "").strip().lower()


def _is_default_text_channel(provider: Optional[str]) -> bool:
    """
    仅依据“配置卡片”判定是否默认通道：
    - provider 在 {"默认", "default", "default_text"} 即视为默认配置卡片
    - 不再依据 api_address
    """
    try:
        p = _normalize_provider(provider)
        return p in {"默认", "default", "default_text"}
    except Exception:
        return False


class RateLimiter:
    """
    基于设备ID和模型的速率限制器
    使用滑动窗口算法实现24小时内的请求次数限制
    """
    
    def __init__(self):
        # 存储格式: {(device_id, model): [(timestamp1, timestamp2, ...)]}
        self._usage_records: Dict[Tuple[str, str], list] = defaultdict(list)
        self._lock = threading.Lock()
        
        # 模型限制配置: {model_name: (max_requests, window_seconds)}
        # 仅对“默认配置卡片”下的指定模型生效；用户自定义渠道不受限
        self._limits = {
            "gemini-2.5-pro": (50, 24 * 3600),
            "gemini-2.5-pro-1M": (50, 24 * 3600),
        }
    
    def check_and_record(
        self,
        device_id: str,
        model: str,
        *,
        api_address: Optional[str] = None,  # 保留参数以兼容调用方，但不参与判断
        provider: Optional[str] = None
    ) -> Tuple[bool, int, int]:
        """
        检查是否允许请求，并记录使用（仅“默认配置卡片”生效）
        
        Args:
            device_id: 设备唯一标识符
            model: 模型名称
            api_address: （忽略）不参与是否默认卡片的判断
            provider: provider/channel（用于判断是否默认卡片）
        """
        # 仅当命中“默认文本通道”且模型在受限清单时限流；否则放行不计数
        if not _is_default_text_channel(provider) or model not in self._limits:
            return True, -1, 0
        
        max_requests, window_seconds = self._limits[model]
        current_time = time.time()
        key = (device_id, model)
        
        with self._lock:
            # 获取该设备+模型的使用记录
            records = self._usage_records[key]
            
            # 清理过期记录（超过时间窗口的记录）
            cutoff_time = current_time - window_seconds
            records[:] = [ts for ts in records if ts > cutoff_time]
            
            # 检查是否超过限制
            current_usage = len(records)
            
            if current_usage >= max_requests:
                # 已达到限制
                oldest_record = min(records) if records else current_time
                reset_time = int(oldest_record + window_seconds)
                remaining = 0
                is_allowed = False
                logger.warning(
                    f"[DEFAULT_LIMIT] Rate limit exceeded for device={device_id[:8]}..., model={model}. "
                    f"Usage: {current_usage}/{max_requests}. Reset at: {reset_time}"
                )
            else:
                # 未达到限制，记录本次使用
                records.append(current_time)
                remaining = max_requests - current_usage - 1
                # 计算重置时间（最早记录的过期时间）
                oldest_record = min(records) if records else current_time
                reset_time = int(oldest_record + window_seconds)
                is_allowed = True
                logger.info(
                    f"[DEFAULT_LIMIT] Rate limit check passed for device={device_id[:8]}..., model={model}. "
                    f"Usage: {current_usage + 1}/{max_requests}, Remaining: {remaining}"
                )
            
            return is_allowed, remaining, reset_time
    
    def get_usage_info(
        self,
        device_id: str,
        model: str,
        *,
        api_address: Optional[str] = None,  # 保留参数以兼容调用方，但不参与判断
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取使用情况信息（不记录使用）
        仅当命中“默认文本通道 + 受限模型”时返回详细限流信息，否则返回未限流。
        """
        if model not in self._limits or not _is_default_text_channel(provider):
            return {
                "limited": False,
                "model": model
            }
        
        max_requests, window_seconds = self._limits[model]
        current_time = time.time()
        key = (device_id, model)
        
        with self._lock:
            records = self._usage_records[key]
            cutoff_time = current_time - window_seconds
            valid_records = [ts for ts in records if ts > cutoff_time]
            current_usage = len(valid_records)
            remaining = max(0, max_requests - current_usage)
            
            oldest_record = min(valid_records) if valid_records else current_time
            reset_time = int(oldest_record + window_seconds)
            
            return {
                "limited": True,
                "model": model,
                "max_requests": max_requests,
                "window_hours": window_seconds / 3600,
                "current_usage": current_usage,
                "remaining": remaining,
                "reset_time": reset_time
            }
    
    def cleanup_old_records(self):
        """
        清理所有过期的记录（可定期调用以释放内存）
        """
        current_time = time.time()
        
        with self._lock:
            keys_to_remove = []
            
            for key, records in self._usage_records.items():
                device_id, model = key
                if model in self._limits:
                    _, window_seconds = self._limits[model]
                    cutoff_time = current_time - window_seconds
                    records[:] = [ts for ts in records if ts > cutoff_time]
                    
                    # 如果记录为空，标记删除
                    if not records:
                        keys_to_remove.append(key)
            
            # 删除空记录
            for key in keys_to_remove:
                del self._usage_records[key]
            
            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} empty rate limit records")


# 全局单例
_rate_limiter_instance = None
_instance_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """获取全局速率限制器实例"""
    global _rate_limiter_instance
    
    if _rate_limiter_instance is None:
        with _instance_lock:
            if _rate_limiter_instance is None:
                _rate_limiter_instance = RateLimiter()
                logger.info("Rate limiter initialized")
    
    return _rate_limiter_instance