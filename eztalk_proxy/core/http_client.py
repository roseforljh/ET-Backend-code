"""
全局 HTTP 客户端管理模块
提供复用的 httpx.AsyncClient 实例，避免频繁创建销毁连接池
"""
import logging
import httpx
from typing import Optional

logger = logging.getLogger("EzTalkProxy.Core.HTTPClient")

# 全局客户端实例
_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    """
    获取全局复用的 HTTP 客户端
    
    配置说明：
    - limits: 连接池限制，max_connections 总连接数，max_keepalive_connections 保持活跃的连接数
    - timeout: 各阶段超时配置
    - follow_redirects: 自动跟随重定向
    - http2: 启用 HTTP/2 支持（如果服务端支持）
    """
    global _http_client
    
    if _http_client is None:
        logger.info("Initializing global HTTP client with connection pooling (optimized for cross-border latency)")
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=200,           # 增加总连接数上限，支持更多并发请求
                max_keepalive_connections=50,  # 增加保持活跃的连接数，减少重建连接开销
                keepalive_expiry=120.0         # 延长保持活跃时间到 120 秒，适合跨境高延迟场景
            ),
            timeout=httpx.Timeout(
                connect=15.0,  # 连接超时，跨境场景适当增加
                read=90.0,     # 读取超时（语音合成可能较长），跨境增加缓冲
                write=15.0,    # 写入超时，跨境增加缓冲
                pool=10.0      # 从连接池获取连接的超时，增加缓冲
            ),
            follow_redirects=True,
            http2=True  # 启用 HTTP/2，多路复用减少连接数
        )
    
    return _http_client

async def close_http_client():
    """
    关闭全局 HTTP 客户端（应用关闭时调用）
    """
    global _http_client
    
    if _http_client is not None:
        logger.info("Closing global HTTP client")
        await _http_client.aclose()
        _http_client = None