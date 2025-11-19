import time
import logging
from datetime import datetime, timedelta, timezone
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from ..core.database import async_session_maker
from ..models.db_models import AccessLog

logger = logging.getLogger("EzTalkProxy.AccessLog")

# 北京时区
BEIJING_TZ = timezone(timedelta(hours=8))

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
                    # 获取真实 IP (处理代理情况)
                    forwarded = request.headers.get("X-Forwarded-For")
                    if forwarded:
                        real_ip = forwarded.split(",")[0].strip()
                    else:
                        real_ip = request.client.host

                    # 使用北京时间
                    current_time_bj = datetime.now(BEIJING_TZ).replace(tzinfo=None) # 移除时区信息以便存入 SQLite

                    access_log = AccessLog(
                        timestamp=current_time_bj,
                        ip_address=real_ip,
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