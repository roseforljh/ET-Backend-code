#!/usr/bin/env python3
"""
EzTalk Proxy 服务启动脚本
"""
import os
import sys
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eztalk_proxy.main import app

logger = logging.getLogger("EzTalkProxy.Runner")

def main():
    """启动EzTalk Proxy服务器"""
    try:
        logger.info("Starting EzTalk Proxy server...")
        
        # 从环境变量获取配置
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "7860"))
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        
        logger.info(f"Host: {host}, Port: {port}")
        logger.info(f"Log Level: {log_level.upper()}")
        
        # 启动服务器
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=True,
            use_colors=True,
            loop="asyncio"
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()