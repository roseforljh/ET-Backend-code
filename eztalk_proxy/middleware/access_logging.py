import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from ..core.database import async_session_maker
from ..models.db_models import AccessLog

logger = logging.getLogger("EzTalkProxy.AccessLog")

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        
        # 异步记录日志到数据库
        # 注意：为了不阻塞主请求，这里应该使用后台任务，但 BaseHTTPMiddleware 不直接支持 BackgroundTasks
        # 简单起见，我们在这里直接 await 写入，或者可以使用 FastAPI 的 BackgroundTasks 在路由中处理
        # 为了性能，这里我们使用 fire-and-forget 模式或者简单的 await (SQLite写入很快)
        
        # 排除健康检查和静态资源
        if request.url.path not in ["/health", "/favicon.ico"] and not request.url.path.startswith("/static"):
             try:
                async with async_session_maker() as session:
                    access_log = AccessLog(
                        ip_address=request.client.host,
                        path=request.url.path,
                        method=request.method,
                        status_code=response.status_code,
                        process_time_ms=process_time,
                        user_agent=request.headers.get("user-agent", "")
                    )
                    session.add(access_log)
                    await session.commit()
             except Exception as e:
                 logger.error(f"Failed to write access log: {e}")

        return response