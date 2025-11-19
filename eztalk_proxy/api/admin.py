import os
import psutil
import time
import hashlib
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Response, status, HTTPException, Request, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy import select, func, text, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.security import verify_admin, create_session_token, COOKIE_NAME, ADMIN_PASSWORD as ENV_ADMIN_PASSWORD
from ..core.config import APP_VERSION
from ..core.logging_utils import memory_log_handler
from ..core.database import get_db
from ..models.db_models import AdminUser, AccessLog

router = APIRouter()

# 启动时间
START_TIME = time.time()

class LoginRequest(BaseModel):
    password: str

class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str

class ConfigUpdateRequest(BaseModel):
    key: str
    value: str

def hash_password(password: str) -> str:
    # 简单哈希，生产环境建议使用 bcrypt
    return hashlib.sha256(password.encode()).hexdigest()

async def get_admin_user(db: AsyncSession) -> Optional[AdminUser]:
    result = await db.execute(select(AdminUser).where(AdminUser.username == "admin"))
    return result.scalars().first()

async def init_admin_user(db: AsyncSession):
    """确保存在 admin 用户"""
    user = await get_admin_user(db)
    if not user:
        # 使用环境变量中的密码初始化
        new_user = AdminUser(
            username="admin",
            hashed_password=hash_password(ENV_ADMIN_PASSWORD)
        )
        db.add(new_user)
        await db.commit()

@router.post("/login")
async def login(request: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """
    管理员登录
    """
    # 确保 admin 用户存在
    await init_admin_user(db)
    user = await get_admin_user(db)
    
    if user and user.hashed_password == hash_password(request.password):
        session_token = create_session_token()
        # 设置 HttpOnly Cookie
        response.set_cookie(
            key=COOKIE_NAME,
            value=session_token,
            httponly=True,
            max_age=86400, # 24小时
            samesite="lax",
            secure=False # 开发环境非HTTPS
        )
        return {"message": "登录成功"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="密码错误"
        )

@router.post("/password", dependencies=[Depends(verify_admin)])
async def change_password(request: PasswordChangeRequest, db: AsyncSession = Depends(get_db)):
    """
    修改管理员密码
    """
    user = await get_admin_user(db)
    if not user:
         raise HTTPException(status_code=500, detail="管理员账户未找到")
         
    if user.hashed_password != hash_password(request.old_password):
         raise HTTPException(status_code=400, detail="旧密码错误")
         
    user.hashed_password = hash_password(request.new_password)
    await db.commit()
    return {"message": "密码修改成功"}

@router.get("/stats", dependencies=[Depends(verify_admin)])
async def get_stats(db: AsyncSession = Depends(get_db)):
    """
    获取系统统计信息
    """
    process = psutil.Process(os.getpid())
    uptime_seconds = time.time() - START_TIME
    uptime_string = str(timedelta(seconds=int(uptime_seconds)))
    
    # 获取今日访问量
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(select(func.count()).select_from(AccessLog).where(AccessLog.timestamp >= today_start))
    today_visits = result.scalar()

    return {
        "app_version": APP_VERSION,
        "uptime": uptime_string,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "process_memory_mb": process.memory_info().rss / 1024 / 1024,
        "connections": len(process.connections()),
        "threads": process.num_threads(),
        "today_visits": today_visits
    }

@router.get("/stats/trend", dependencies=[Depends(verify_admin)])
async def get_stats_trend(period: str = "day", db: AsyncSession = Depends(get_db)):
    """
    获取访问趋势数据
    period: day (按小时), month (按天), year (按月)
    """
    now = datetime.utcnow()
    
    if period == "day":
        start_time = now - timedelta(hours=24)
        # SQLite 的 strftime 格式
        group_format = "%Y-%m-%d %H:00" 
        interval = timedelta(hours=1)
    elif period == "month":
        start_time = now - timedelta(days=30)
        group_format = "%Y-%m-%d"
        interval = timedelta(days=1)
    elif period == "year":
        start_time = now - timedelta(days=365)
        group_format = "%Y-%m"
        interval = timedelta(days=30) # 近似
    else:
        raise HTTPException(status_code=400, detail="Invalid period")

    # 使用 SQLAlchemy 构建查询
    # 注意：SQLite 的日期函数
    stmt = (
        select(
            func.strftime(group_format, AccessLog.timestamp).label('time_bucket'),
            func.count().label('count')
        )
        .where(AccessLog.timestamp >= start_time)
        .group_by('time_bucket')
        .order_by('time_bucket')
    )
    
    result = await db.execute(stmt)
    data = result.all()
    
    return [{"time": row.time_bucket, "count": row.count} for row in data]

@router.get("/stats/access-logs", dependencies=[Depends(verify_admin)])
async def get_access_logs(
    limit: int = 100,
    offset: int = 0,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    keyword: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    获取详细访问日志，支持筛选
    """
    stmt = select(AccessLog).order_by(AccessLog.timestamp.desc())
    
    if start_time:
        stmt = stmt.where(AccessLog.timestamp >= start_time)
    
    if end_time:
        stmt = stmt.where(AccessLog.timestamp <= end_time)
        
    if keyword:
        search_pattern = f"%{keyword}%"
        stmt = stmt.where(
            or_(
                AccessLog.ip_address.ilike(search_pattern),
                AccessLog.path.ilike(search_pattern),
                AccessLog.method.ilike(search_pattern)
            )
        )
        
    stmt = stmt.limit(limit).offset(offset)
    
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return logs

@router.get("/logs", dependencies=[Depends(verify_admin)])
async def get_logs(limit: int = 100):
    """
    获取最近的内存日志
    """
    return memory_log_handler.get_logs(limit)

@router.get("/config", dependencies=[Depends(verify_admin)])
async def get_config():
    """
    获取当前环境变量配置（仅显示 .env 文件中的内容）
    """
    env_path = ".env"
    config_items = []
    
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key, value = parts
                        # 隐藏敏感信息
                        if "KEY" in key or "SECRET" in key or "PASSWORD" in key:
                            value = "******" + value[-4:] if len(value) > 4 else "******"
                        config_items.append({"key": key, "value": value})
    
    return config_items

@router.post("/config", dependencies=[Depends(verify_admin)])
async def update_config(request: ConfigUpdateRequest):
    """
    更新配置（写入 .env 文件）
    """
    env_path = ".env"
    lines = []
    key_found = False
    
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{request.key}="):
            new_lines.append(f"{request.key}={request.value}\n")
            key_found = True
        else:
            new_lines.append(line)
            
    if not key_found:
        new_lines.append(f"\n{request.key}={request.value}\n")
        
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    # 更新当前进程的环境变量（部分配置可能需要重启生效）
    os.environ[request.key] = request.value
    
    return {"message": f"配置 '{request.key}' 已更新"}

@router.post("/restart", dependencies=[Depends(verify_admin)])
async def restart_service():
    """
    尝试重启服务
    """
    import threading
    
    def kill_server():
        time.sleep(1)
        os._exit(0)
        
    threading.Thread(target=kill_server).start()
    return {"message": "服务正在重启..."}