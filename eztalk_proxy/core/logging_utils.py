import logging
import collections
from datetime import datetime

class MemoryLogHandler(logging.Handler):
    """
    将日志记录到内存中的 Handler，用于在管理界面显示
    """
    def __init__(self, capacity=1000):
        super().__init__()
        self.log_buffer = collections.deque(maxlen=capacity)
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_buffer.append({
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(), # 获取原始消息，不含格式化
                "formatted": msg # 完整格式化消息
            })
        except Exception:
            self.handleError(record)

    def get_logs(self, limit=100):
        """获取最近的日志"""
        return list(self.log_buffer)[-limit:]

# 全局实例
memory_log_handler = MemoryLogHandler()