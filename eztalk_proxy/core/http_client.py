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
        logger.info("Initializing global HTTP client with connection pooling")
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=100,  # 总连接数上限
                max_keepalive_connections=20,  # 保持活跃的连接数
                keepalive_expiry=30.0  # 保持活跃时间（秒）
            ),
            timeout=httpx.Timeout(
                connect=10.0,  # 连接超时
                read=60.0,     # 读取超时（语音合成可能较长）
                write=10.0,    # 写入超时
                pool=5.0       # 从连接池获取连接的超时
            ),
            follow_redirects=True,
            http2=True  # 启用 HTTP/2
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