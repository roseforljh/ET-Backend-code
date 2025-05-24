import os
import logging
import httpx
from fastapi import FastAPI, Request # <--- 确保 Request 在这里导入
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from .config import (
    APP_VERSION, API_TIMEOUT, READ_TIMEOUT, MAX_CONNECTIONS,
    LOG_LEVEL_FROM_ENV
)
from .routers import chat as chat_router

numeric_level = getattr(logging, LOG_LEVEL_FROM_ENV, logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EzTalkProxy.Main")

logging.getLogger("EzTalkProxy.SPHASANN").setLevel(LOG_LEVEL_FROM_ENV)

for lib_logger_name in ["httpx", "httpcore", "googleapiclient.discovery_cache", "uvicorn.access"]:
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global http_client
    logger.info("Lifespan: 初始化HTTP客户端...")
    client_local: Optional[httpx.AsyncClient] = None
    try:
        client_local = httpx.AsyncClient(
            timeout=httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
            http2=True,
            follow_redirects=True,
            trust_env=False
        )
        app_instance.state.http_client = client_local
        http_client = client_local
        logger.info("Lifespan: HTTP客户端初始化成功。")
    except Exception as e:
        logger.error(f"Lifespan: HTTP客户端初始化失败: {e}", exc_info=True)
        app_instance.state.http_client = None
        http_client = None

    yield

    logger.info("Lifespan: 关闭HTTP客户端...")
    client_to_close = getattr(app_instance.state, "http_client", None)
    if client_to_close:
        try:
            await client_to_close.aclose()
        except Exception as e:
            logger.error(f"Lifespan: 关闭HTTP客户端错误: {e}", exc_info=True)
    app_instance.state.http_client = None
    http_client = None
    logger.info("Lifespan: 关闭完成。")

app = FastAPI(
    title="EzTalk Proxy",
    description=f"代理服务，版本: {APP_VERSION}",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
logger.info(f"FastAPI EzTalk Proxy v{APP_VERSION} 初始化完成，已配置CORS。")

app.include_router(chat_router.router) 
@app.get("/health", status_code=200, include_in_schema=False, tags=["Utilities"])
async def health_check(request: Request):
    logger.info("Health check endpoint called.") # 添加日志
    try:
        client_from_state = getattr(request.app.state, "http_client", "NOT_FOUND_IN_STATE") # 更明确的默认值
        client_is_closed = "UNKNOWN"
        if hasattr(client_from_state, 'is_closed'):
            client_is_closed = client_from_state.is_closed
        
        logger.info(f"Health check: client_from_state type: {type(client_from_state)}, is_closed: {client_is_closed}")

        client_status = "ok" if client_from_state != "NOT_FOUND_IN_STATE" and hasattr(client_from_state, 'is_closed') and not client_from_state.is_closed else "warning"
        detail_message = f"HTTP client {'initialized and open' if client_status == 'ok' else ('not initialized or closed' if client_from_state == 'NOT_FOUND_IN_STATE' or not hasattr(client_from_state, 'is_closed') else 'closed or uninitialized')}"
        
        response_data = {"status": client_status, "detail": detail_message}
        logger.info(f"Health check response: {response_data}")
        return response_data
    except Exception as e:
        logger.error(f"Error in health check endpoint: {e}", exc_info=True)
        return {"status": "error_in_health_check", "detail": str(e)} # 仍然返回 200，但内容表明问题


if __name__ == "__main__":
    import uvicorn
    # Request 已经移到顶部导入了

    APP_HOST = os.getenv("HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("PORT", 8000))
    DEV_RELOAD = os.getenv("DEV_RELOAD", "false").lower() == "true"

    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"].setdefault("default", {"fmt": "", "datefmt": "", "use_colors": None})
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)-8s [%(name)s] - %(message)s"
    log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["formatters"].setdefault("access", {"fmt": "", "datefmt": "", "use_colors": None})
    log_config["formatters"]["access"]["fmt"] = '%(asctime)s %(levelname)-8s [%(name)s] - %(client_addr)s - "%(request_line)s" %(status_code)s'
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    log_config["handlers"].setdefault("default", {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"})
    log_config["handlers"].setdefault("access", {"formatter": "access", "class": "logging.StreamHandler", "stream": "ext://sys.stdout"})
    log_config.setdefault("loggers", {})
    log_config["loggers"]["uvicorn.error"] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    log_config["loggers"]["uvicorn.access"] = {"handlers": ["access"], "level": "WARNING", "propagate": False}

    logger.info(f"Starting Uvicorn server: http://{APP_HOST}:{APP_PORT}")
    logger.info(f"Development reload: {DEV_RELOAD}")
    logger.info(f"Application Log Level (EzTalkProxy.*): {LOG_LEVEL_FROM_ENV}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_config=log_config, reload=DEV_RELOAD)