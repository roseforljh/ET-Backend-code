import os
import logging
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional

from .core.config import (
    APP_VERSION, API_TIMEOUT, READ_TIMEOUT, MAX_CONNECTIONS,
    LOG_LEVEL_FROM_ENV,
    TEMP_UPLOAD_DIR
)
from .api import chat as chat_router
from .api import image_generation as image_generation_router
from .api import seedream as seedream_router
from .api import gemini_live as gemini_live_router

numeric_log_level = getattr(logging, LOG_LEVEL_FROM_ENV.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_log_level,
    format='%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EzTalkProxy.Main")

if hasattr(logging.getLogger("EzTalkProxy"), 'SPHASANN'):
    logging.getLogger("EzTalkProxy.SPHASANN").setLevel(LOG_LEVEL_FROM_ENV.upper())

for lib_logger_name in ["httpx", "httpcore", "googleapiclient.discovery_cache", "uvicorn.access", "watchfiles"]:
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Lifespan: 应用启动，开始初始化...")
    client_local: Optional[httpx.AsyncClient] = None
    try:
        client_local = httpx.AsyncClient(
            timeout=httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT),
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
            http2=True,
            follow_redirects=True,
            trust_env=True
        )
        app_instance.state.http_client = client_local
        logger.info(f"Lifespan: HTTP客户端初始化成功。Timeout Connect: {API_TIMEOUT}s, Read Timeout: {READ_TIMEOUT}s, Max Connections: {MAX_CONNECTIONS}")

        if not os.path.exists(TEMP_UPLOAD_DIR):
            try:
                os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
                logger.info(f"Lifespan: 成功创建或已存在临时上传目录: {TEMP_UPLOAD_DIR}")
            except OSError as e_mkdir:
                logger.error(f"Lifespan: 创建临时上传目录 {TEMP_UPLOAD_DIR} 失败: {e_mkdir}", exc_info=True)
        else:
            logger.info(f"Lifespan: 临时上传目录已存在: {TEMP_UPLOAD_DIR}")

    except Exception as e:
        logger.error(f"Lifespan: HTTP客户端初始化过程中发生错误: {e}", exc_info=True)
        app_instance.state.http_client = None
    
    yield

    logger.info("Lifespan: 应用关闭，开始关闭HTTP客户端...")
    client_to_close = getattr(app_instance.state, "http_client", None)
    if client_to_close and isinstance(client_to_close, httpx.AsyncClient) and not client_to_close.is_closed:
        try:
            await client_to_close.aclose()
            logger.info("Lifespan: HTTP客户端成功关闭。")
        except Exception as e:
            logger.error(f"Lifespan: 关闭HTTP客户端时发生错误: {e}", exc_info=True)
    elif client_to_close and isinstance(client_to_close, httpx.AsyncClient) and client_to_close.is_closed:
        logger.info("Lifespan: HTTP客户端先前已经关闭。")
    else:
        logger.warning("Lifespan: HTTP客户端未找到、状态未知或类型不正确，可能无需关闭或已处理。")
    
    if hasattr(app_instance.state, "http_client"):
        delattr(app_instance.state, "http_client")

    logger.info("Lifespan: 应用关闭流程完成。")


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
logger.info("聊天路由已加载到路径 /api/v1/chat (或其他在chat_router中定义的路径)")

app.include_router(image_generation_router.router)
logger.info("图像生成路由已加载到路径 /images/generations")

# Doubao Seedream 4.0 image generation proxy
app.include_router(seedream_router.router)
logger.info("Doubao Seedream 路由已加载到路径 /doubao/v3/images/generations")

# Gemini Live audio relay (server-to-server)
app.include_router(gemini_live_router.router)
logger.info("Gemini Live 路由已加载到路径 /gemini/live/*")


@app.get("/", status_code=200, include_in_schema=False, tags=["Utilities"])
async def root():
    """根路由，确认服务正常运行"""
    return {
        "message": "EzTalk Proxy API is running",
        "version": APP_VERSION,
        "status": "ok",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", status_code=200, include_in_schema=False, tags=["Utilities"])
async def health_check(request: Request):
    client_from_state = getattr(request.app.state, "http_client", None)
    client_status = "ok"
    detail_message = "HTTP client initialized and seems operational."

    if client_from_state is None:
        client_status = "error"
        detail_message = "HTTP client not initialized in app.state."
    elif not isinstance(client_from_state, httpx.AsyncClient):
        client_status = "error"
        detail_message = f"Unexpected object type in app.state.http_client: {type(client_from_state)}"
    elif client_from_state.is_closed:
        client_status = "warning"
        detail_message = "HTTP client in app.state is closed."

    response_data = {"status": client_status, "detail": detail_message, "app_version": APP_VERSION}
    return response_data


# This block is now handled by the top-level run.py script