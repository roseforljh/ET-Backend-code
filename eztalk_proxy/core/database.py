from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
import os

# 数据库文件路径
DATABASE_URL = "sqlite+aiosqlite:///./eztalk.db"

# 创建异步引擎
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    poolclass=NullPool
)

# 创建 Session 工厂
async_session_maker = sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    """
    依赖项：获取数据库会话
    """
    async with async_session_maker() as session:
        yield session

async def init_db():
    """
    初始化数据库表
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)