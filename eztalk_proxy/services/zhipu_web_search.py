"""
智谱Web Search API服务
"""
import asyncio
import logging
from typing import List, Dict, Optional, Any
import httpx
from datetime import datetime

logger = logging.getLogger("EzTalkProxy.ZhipuWebSearch")

class ZhipuWebSearchService:
    """智谱Web Search API服务"""
    
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
    
    async def search(
        self,
        api_key: str,
        search_query: str,
        request_id: str,
        search_engine: str = "search_pro",
        search_intent: bool = False,
        count: int = 5,
        search_domain_filter: Optional[str] = None,
        search_recency_filter: str = "year",
        content_size: str = "high"
    ) -> Dict[str, Any]:
        """
        调用智谱Web Search API
        
        Args:
            api_key: 智谱API密钥
            search_query: 搜索内容
            request_id: 请求ID
            search_engine: 搜索引擎类型
            search_intent: 是否进行搜索意图识别
            count: 返回结果条数
            search_domain_filter: 域名过滤
            search_recency_filter: 时间范围过滤
            content_size: 内容长度控制
            
        Returns:
            搜索结果
        """
        log_prefix = f"RID-{request_id}"
        
        # 构建请求体
        payload = {
            "search_query": search_query[:70],  # 限制最大70字符
            "search_engine": search_engine,
            "search_intent": search_intent,
            "count": min(count, 50),  # 最大50条
            "search_recency_filter": search_recency_filter,
            "content_size": content_size,
            "request_id": request_id
        }
        
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"{log_prefix}: Calling Zhipu Web Search API with query: '{search_query[:50]}...', engine: {search_engine}")
        
        try:
            response = await self.http_client.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"{log_prefix}: Zhipu Web Search successful, got {len(result.get('search_result', []))} results")
                return result
            else:
                logger.error(f"{log_prefix}: Zhipu Web Search failed with status {response.status_code}: {response.text}")
                return {
                    "search_intent": [],
                    "search_result": [],
                    "error": f"API returned status {response.status_code}"
                }
                
        except httpx.TimeoutException:
            logger.error(f"{log_prefix}: Zhipu Web Search timeout")
            return {
                "search_intent": [],
                "search_result": [],
                "error": "Request timeout"
            }
        except Exception as e:
            logger.error(f"{log_prefix}: Zhipu Web Search error: {e}")
            return {
                "search_intent": [],
                "search_result": [],
                "error": str(e)
            }
    
    def format_search_results_for_llm(self, search_result: List[Dict[str, Any]]) -> str:
        """
        格式化搜索结果供LLM使用
        
        Args:
            search_result: 搜索结果列表
            
        Returns:
            格式化的文本
        """
        if not search_result:
            return "未找到相关搜索结果。"
        
        formatted_results = []
        current_date = datetime.now().strftime("%Y年%m月%d日")
        
        formatted_results.append(f"[搜索时间: {current_date}]\n")
        
        for i, result in enumerate(search_result, 1):
            title = result.get("title", "")
            content = result.get("content", "")
            link = result.get("link", "")
            media = result.get("media", "")
            publish_date = result.get("publish_date", "")
            
            formatted_item = f"{i}. **{title}**"
            if media:
                formatted_item += f" - {media}"
            if publish_date:
                formatted_item += f" ({publish_date})"
            formatted_item += f"\n   {content}"
            if link:
                formatted_item += f"\n   来源: {link}"
            
            formatted_results.append(formatted_item)
        
        return "\n\n".join(formatted_results)
    
    async def search_and_format(
        self,
        api_key: str,
        search_query: str,
        request_id: str,
        **kwargs
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        搜索并格式化结果
        
        Returns:
            (格式化的文本, 原始搜索结果列表)
        """
        result = await self.search(
            api_key=api_key,
            search_query=search_query,
            request_id=request_id,
            **kwargs
        )
        
        search_results = result.get("search_result", [])
        formatted_text = self.format_search_results_for_llm(search_results)
        
        # 转换为前端期望的格式
        web_search_results = []
        for item in search_results:
            web_search_results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("content", "")
            })
        
        return formatted_text, web_search_results