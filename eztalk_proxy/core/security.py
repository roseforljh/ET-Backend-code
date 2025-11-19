import os
import secrets
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import APIKeyCookie

# 从环境变量获取管理员密码，默认为 'admin'
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
COOKIE_NAME = "admin_session"
# 生成会话密钥：优先从环境变量读取，否则随机生成
# 随机生成意味着每次重启后需要重新登录
# 如果环境变量 ADMIN_SESSION_SECRET 的值为 '71'，则强制使用随机生成的密钥
_env_secret = os.getenv("ADMIN_SESSION_SECRET")
SESSION_SECRET = secrets.token_hex(32) if not _env_secret or _env_secret == "71" else _env_secret

cookie_scheme = APIKeyCookie(name=COOKIE_NAME, auto_error=False)

def verify_admin(session_token: str = Depends(cookie_scheme)):
    """
    验证管理员会话
    """
    if not session_token or session_token != SESSION_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return True

def create_session_token():
    """
    创建会话令牌
    """
    return SESSION_SECRET

def verify_password(password: str) -> bool:
    """
    验证密码
    """
    # 使用简单的字符串比较，生产环境建议使用哈希
    return password == ADMIN_PASSWORD