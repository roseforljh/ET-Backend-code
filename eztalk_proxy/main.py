import os
import logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
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
# from .api import voice_chat as voice_chat_router
# from .api import voice_chat_ws as voice_chat_ws_router
from .api import admin as admin_router
from .middleware import SignatureVerificationMiddleware
from .middleware.access_logging import AccessLogMiddleware
from .core.logging_utils import memory_log_handler
from .core.database import init_db

numeric_log_level = getattr(logging, LOG_LEVEL_FROM_ENV.upper(), logging.INFO)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(numeric_log_level)

# 1. 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)-8s [%(name)s:%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
root_logger.addHandler(console_handler)

# 2. 内存处理器 (用于管理后台)
root_logger.addHandler(memory_log_handler)

logger = logging.getLogger("EzTalkProxy.Main")

if hasattr(logging.getLogger("EzTalkProxy"), 'SPHASANN'):
    logging.getLogger("EzTalkProxy.SPHASANN").setLevel(LOG_LEVEL_FROM_ENV.upper())

for lib_logger_name in ["httpx", "httpcore", "googleapiclient.discovery_cache", "uvicorn.access", "watchfiles"]:
    logging.getLogger(lib_logger_name).setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("Lifespan: 应用启动，开始初始化...")
    
    # 初始化数据库
    try:
        await init_db()
        logger.info("Lifespan: 数据库初始化成功")
    except Exception as e:
        logger.error(f"Lifespan: 数据库初始化失败: {e}", exc_info=True)

    client_local: Optional[httpx.AsyncClient] = None
    try:
        client_local = httpx.AsyncClient(
            timeout=httpx.Timeout(API_TIMEOUT, read=READ_TIMEOUT),
            limits=httpx.Limits(
                max_connections=MAX_CONNECTIONS,
                max_keepalive_connections=50,  # 保持活跃的连接数，减少重新建立连接的开销
                keepalive_expiry=120.0         # 连接保持活跃时间（秒），适合跨境高延迟场景
            ),
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

# ===== 添加签名验证中间件（必须在CORS之前，在路由注册之前）=====
# 从环境变量读取配置
signature_enabled = os.getenv("SIGNATURE_VERIFICATION_ENABLED", "false").lower() == "true"
signature_keys_str = os.getenv("SIGNATURE_SECRET_KEYS", "your-secret-key-change-in-production-2024")
signature_keys = [key.strip() for key in signature_keys_str.split(",") if key.strip()]

# 只有当签名验证启用时才添加中间件
# 注意：中间件的执行顺序是后添加先执行，所以签名验证要在CORS之前添加
if signature_enabled:
    app.add_middleware(
        SignatureVerificationMiddleware,
        secret_keys=signature_keys,
        signature_validity_seconds=300,  # 5分钟
        excluded_paths=["/health", "/docs", "/redoc", "/openapi.json", "/", "/everytalk", "/favicon.ico"],
        enabled=True  # 强制为True，因为我们只在启用时添加
    )
    logger.info("签名验证中间件已启用")
    logger.info(f"排除路径: {['/health', '/docs', '/redoc', '/openapi.json', '/', '/everytalk', '/favicon.ico']}")
    logger.info(f"密钥数量: {len(signature_keys)}")
    logger.info(f"签名有效期: 300秒")
else:
    logger.warning("签名验证中间件已禁用（开发模式）")

# 访问日志中间件
app.add_middleware(AccessLogMiddleware)

# CORS中间件必须在签名验证之后添加（这样CORS会先执行）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# GZip 压缩中间件：减少 JSON 响应传输量，优化跨境延迟
# minimum_size=500: 只压缩大于 500 字节的响应
app.add_middleware(GZipMiddleware, minimum_size=500)
logger.info(f"FastAPI EzTalk Proxy v{APP_VERSION} 初始化完成，已配置CORS。")

# ===== 注册路由（必须在所有中间件之后）=====

app.include_router(chat_router.router)
logger.info("聊天路由已加载到路径 /api/v1/chat (或其他在chat_router中定义的路径)")

app.include_router(image_generation_router.router)
logger.info("图像生成路由已加载到路径 /images/generations")

# Doubao Seedream 4.0 image generation proxy
app.include_router(seedream_router.router)
logger.info("Doubao Seedream 路由已加载到路径 /doubao/v3/images/generations")

# Voice Chat (STT + Chat + TTS) - 已废弃，客户端改为直连
# app.include_router(voice_chat_router.router, prefix="/voice-chat")
# logger.info("Voice Chat 路由已加载到路径 /voice-chat/*")

# Voice Chat WebSocket (实时流式对话) - 已废弃，客户端改为直连
# app.include_router(voice_chat_ws_router.router, prefix="/voice-chat")
# logger.info("Voice Chat WebSocket 路由已加载到路径 /voice-chat/realtime")

# Admin Router
app.include_router(admin_router.router, prefix="/everytalk/api", tags=["Admin"])
logger.info("管理后台 API 已加载到路径 /everytalk/api")

@app.get("/everytalk", response_class=HTMLResponse, include_in_schema=False)
async def admin_page():
    """管理后台页面"""
    # 假设模板文件在 eztalk_proxy/templates/admin.html
    # 在生产环境中，应该使用 Jinja2Templates 或者正确配置静态文件路径
    # 这里为了简化部署，直接读取文件内容返回
    import os
    template_path = os.path.join(os.path.dirname(__file__), "templates", "admin.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Admin template not found."

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