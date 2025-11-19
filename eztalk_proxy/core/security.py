import os
import secrets
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import APIKeyCookie

# 从环境变量获取管理员密码，默认为 'admin'
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
COOKIE_NAME = "admin_session"
# 生成一个随机的会话密钥，每次重启都会变化，这意味重启后需要重新登录
# 在生产环境中，应该将其持久化或使用 JWT
SESSION_SECRET = secrets.token_hex(32)

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