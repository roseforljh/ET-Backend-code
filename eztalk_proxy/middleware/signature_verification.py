"""
请求签名验证中间件
用于验证客户端请求的签名，防止API滥用和中间人攻击
"""
import hmac
import hashlib
import base64
import time
import logging
from typing import Optional, List
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.datastructures import Headers
import io

logger = logging.getLogger("EzTalkProxy.SignatureMiddleware")

class SignatureVerificationMiddleware(BaseHTTPMiddleware):
    """
    签名验证中间件
    
    验证流程：
    1. 检查请求头中是否包含 X-Signature 和 X-Timestamp
    2. 验证时间戳是否在有效期内（防重放攻击）
    3. 使用相同算法计算签名并比较
    4. 验证通过后继续处理请求
    """
    
    def __init__(
        self,
        app,
        secret_keys: List[str],
        signature_validity_seconds: int = 300,  # 5分钟
        excluded_paths: Optional[List[str]] = None,
        enabled: bool = True
    ):
        """
        初始化签名验证中间件
        
        Args:
            app: FastAPI应用实例
            secret_keys: 签名密钥列表（支持多个密钥用于密钥轮换）
            signature_validity_seconds: 签名有效期（秒）
            excluded_paths: 排除验证的路径列表（如健康检查端点）
            enabled: 是否启用签名验证
        """
        super().__init__(app)
        self.secret_keys = secret_keys
        self.signature_validity_seconds = signature_validity_seconds
        self.excluded_paths = excluded_paths or ["/health", "/docs", "/redoc", "/openapi.json", "/", "/everytalk", "/favicon.ico"]
        self.enabled = enabled
        
        if not self.enabled:
            logger.warning("签名验证中间件已禁用")
        else:
            logger.info(f"签名验证中间件已启用，有效期: {signature_validity_seconds}秒")
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        
        # 如果中间件未启用，直接放行
        if not self.enabled:
            logger.debug(f"签名验证已禁用，放行请求: {request.method} {request.url.path}")
            return await call_next(request)
        
        # 检查是否是排除路径
        if self._is_excluded_path(request.url.path):
            logger.debug(f"排除路径，无需验证: {request.method} {request.url.path}")
            return await call_next(request)
            
        # 额外检查：如果路径以 /everytalk 开头，也跳过验证
        # 这样可以确保 /everytalk/api/* 等子路径也被排除
        if request.url.path.startswith("/everytalk"):
             logger.debug(f"管理后台路径，无需验证: {request.method} {request.url.path}")
             return await call_next(request)
        
        # 记录开始验证
        logger.info(f"🔐 开始签名验证: {request.method} {request.url.path}")
        
        try:
            # 读取并缓存请求体
            body = await request.body()
            
            # 验证签名
            await self._verify_signature_with_body(request, body)
            
            # 签名验证通过，继续处理请求
            # 不需要修改 receive,因为 request.body() 已经缓存了请求体
            # FastAPI/Starlette 会自动处理后续的请求体读取
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # 签名验证失败，返回错误响应
            logger.warning(
                f"签名验证失败: {e.detail} | "
                f"Path: {request.url.path} | "
                f"Method: {request.method} | "
                f"Client: {request.client.host if request.client else 'unknown'}"
            )
            return Response(
                content=f'{{"detail": "{e.detail}"}}',
                status_code=e.status_code,
                media_type="application/json"
            )
        except Exception as e:
            logger.error(f"签名验证过程中发生错误: {str(e)}", exc_info=True)
            return Response(
                content='{"detail": "Internal server error during signature verification"}',
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                media_type="application/json"
            )
    
    def _is_excluded_path(self, path: str) -> bool:
        """检查路径是否在排除列表中"""
        # 注意：不能简单使用 startswith，因为 "/" 会匹配所有路径
        # 需要精确匹配或者匹配特定前缀
        for excluded in self.excluded_paths:
            if excluded == "/":
                # 根路径只匹配精确的 "/"
                if path == "/":
                    return True
            elif path.startswith(excluded):
                # 其他路径使用 startswith 匹配
                return True
        return False
    
    async def _verify_signature_with_body(self, request: Request, body: bytes):
        """验证请求签名（使用已读取的请求体）"""
        
        # 1. 获取签名和时间戳
        signature = request.headers.get("X-Signature")
        timestamp_str = request.headers.get("X-Timestamp")
        
        if not signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing signature header (X-Signature)"
            )
        
        if not timestamp_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing timestamp header (X-Timestamp)"
            )
        
        # 2. 验证时间戳
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid timestamp format"
            )
        
        current_time = int(time.time() * 1000)  # 毫秒
        time_diff = abs(current_time - timestamp)
        
        if time_diff > (self.signature_validity_seconds * 1000):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Signature expired (time diff: {time_diff}ms)"
            )
        
        # 3. 使用传入的请求体
        # 对于 multipart/form-data 请求,使用空字符串计算签名
        # 因为 multipart 的边界和编码在客户端和服务端可能不同
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type.lower():
            body_str = ""
            logger.debug(f"检测到 multipart/form-data 请求,使用空字符串计算签名")
        else:
            body_str = body.decode('utf-8') if body else ""
        
        # 4. 计算签名
        method = request.method.upper()
        path = request.url.path
        
        expected_signature = self._calculate_signature(
            method=method,
            path=path,
            body=body_str,
            timestamp=timestamp
        )
        
        # 5. 比较签名（尝试所有配置的密钥）
        signature_valid = False
        for expected_sig in expected_signature:
            if hmac.compare_digest(signature, expected_sig):
                signature_valid = True
                break
        
        if not signature_valid:
            logger.warning(f"❌ 签名验证失败: {method} {path} | 提供的签名: {signature[:20]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )
        
        logger.info(f"✅ 签名验证成功: {method} {path}")
    
    def _calculate_signature(
        self,
        method: str,
        path: str,
        body: str,
        timestamp: int
    ) -> List[str]:
        """
        计算请求签名（使用所有配置的密钥）
        
        Args:
            method: HTTP方法
            path: 请求路径
            body: 请求体
            timestamp: 时间戳
            
        Returns:
            签名列表（每个密钥对应一个签名）
        """
        # 1. 计算请求体的SHA-256哈希
        if body:
            body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
        else:
            body_hash = ""
        
        # 2. 构建待签名字符串
        # 格式: timestamp|method|path|bodyHash
        signature_data = f"{timestamp}|{method}|{path}|{body_hash}"
        
        # 3. 使用每个密钥计算HMAC-SHA256签名
        signatures = []
        for secret_key in self.secret_keys:
            hmac_obj = hmac.new(
                secret_key.encode('utf-8'),
                signature_data.encode('utf-8'),
                hashlib.sha256
            )
            signature = base64.b64encode(hmac_obj.digest()).decode('ascii')
            signatures.append(signature)
        
        return signatures


def create_signature_middleware(
    secret_keys: Optional[List[str]] = None,
    signature_validity_seconds: int = 300,
    excluded_paths: Optional[List[str]] = None,
    enabled: bool = True
) -> SignatureVerificationMiddleware:
    """
    创建签名验证中间件的工厂函数
    
    Args:
        secret_keys: 签名密钥列表
        signature_validity_seconds: 签名有效期（秒）
        excluded_paths: 排除验证的路径列表
        enabled: 是否启用
        
    Returns:
        SignatureVerificationMiddleware实例
    """
    import os
    
    # 从环境变量获取密钥
    if secret_keys is None:
        env_keys = os.getenv("SIGNATURE_SECRET_KEYS", "")
        if env_keys:
            secret_keys = [key.strip() for key in env_keys.split(",") if key.strip()]
        else:
            # 默认密钥（仅用于开发环境）
            secret_keys = ["your-secret-key-change-in-production-2024"]
            logger.warning("使用默认签名密钥，生产环境请配置 SIGNATURE_SECRET_KEYS 环境变量")
    
    # 从环境变量获取是否启用
    if os.getenv("SIGNATURE_VERIFICATION_ENABLED", "").lower() == "false":
        enabled = False
    
    return lambda app: SignatureVerificationMiddleware(
        app=app,
        secret_keys=secret_keys,
        signature_validity_seconds=signature_validity_seconds,
        excluded_paths=excluded_paths,
        enabled=enabled
    )