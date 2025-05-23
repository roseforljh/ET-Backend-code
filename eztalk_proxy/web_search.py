import asyncio
import logging
import orjson
from typing import List, Dict

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import GOOGLE_API_KEY_ENV, GOOGLE_CSE_ID, SEARCH_RESULT_COUNT, SEARCH_SNIPPET_MAX_LENGTH

logger = logging.getLogger("EzTalkProxy.WebSearch")

async def perform_web_search(query: str, rid: str) -> List[Dict[str, str]]:
    results = []
    actual_google_api_key = GOOGLE_API_KEY_ENV 
    if not actual_google_api_key or not GOOGLE_CSE_ID:
        logger.warning(f"RID-{rid}: Web search skipped, GOOGLE_API_KEY_ENV or GOOGLE_CSE_ID not set.")
        return results
    if not query:
        logger.warning(f"RID-{rid}: Web search skipped, query is empty.")
        return results

    try:
        def search_sync():
            service = build("customsearch", "v1", developerKey=actual_google_api_key, cache_discovery=False)
            res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=min(SEARCH_RESULT_COUNT, 10)).execute()
            return res.get('items', [])

        logger.info(f"RID-{rid}: Performing web search for query: '{query[:100]}'")
        search_items = await asyncio.to_thread(search_sync)

        for i, item in enumerate(search_items):
            snippet = item.get('snippet', 'N/A').replace('\n', ' ').strip()
            if len(snippet) > SEARCH_SNIPPET_MAX_LENGTH:
                snippet = snippet[:SEARCH_SNIPPET_MAX_LENGTH] + "..."
            results.append({
                "index": i + 1,
                "title": item.get('title', 'N/A').strip(),
                "href": item.get('link', 'N/A'),
                "snippet": snippet
            })
        logger.info(f"RID-{rid}: Web search completed, found {len(results)} results.")

    except HttpError as e:
        err_content = "Unknown Google API error"
        status_code = "N/A"
        if hasattr(e, 'resp') and hasattr(e.resp, 'status'):
            status_code = e.resp.status
        try:
            content_json = orjson.loads(e.content)
            err_detail = content_json.get("error", {})
            err_message = err_detail.get("message", str(e.content)) 
            err_content = f"{err_message} (Code: {err_detail.get('code', 'N/A')}, Status: {err_detail.get('status', 'N/A')})"
        except: 
            err_content = e._get_reason() if hasattr(e, '_get_reason') else e.content.decode(errors='ignore')[:200]
        logger.error(f"RID-{rid}: Google Web Search HttpError (Status: {status_code}): {err_content}")
    except Exception as search_exc:
        logger.error(f"RID-{rid}: Google Web Search failed for query '{query[:50]}': {search_exc}", exc_info=True)
    return results

def generate_search_context_message_content(query: str, search_results: List[Dict[str, str]]) -> str:
    if not search_results:
        return ""
    
    parts = [f"Web search results for the query '{query}':"]
    for res in search_results:
        parts.append(f"\n[{res.get('index')}] Title: {res.get('title')}\n     Snippet: {res.get('snippet')}\n     Source URL (for AI reference only, do not cite directly): {res.get('href')}")
    
    return "\n".join(parts) + f"\n\nPlease use these search results to answer the user's query. Incorporate information from these results as much as possible into your response. Cite search results using the format [index] if you use their information."