from sqlalchemy import Column, Integer, String, DateTime, Float
from datetime import datetime
from ..core.database import Base

class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class AccessLog(Base):
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String, index=True)
    path = Column(String, index=True)
    method = Column(String)
    status_code = Column(Integer)
    process_time_ms = Column(Float)
    user_agent = Column(String)