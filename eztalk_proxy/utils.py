import orjson
import re
import logging
import datetime
from typing import Any, Dict, List, Tuple, Optional
from fastapi.responses import JSONResponse

from .config import COMMON_HEADERS, MAX_SSE_LINE_LENGTH

logger = logging.getLogger("EzTalkProxy.Utils")


def orjson_dumps_bytes_wrapper(data: Any) -> bytes:
    return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_PASSTHROUGH_DATETIME | orjson.OPT_APPEND_NEWLINE)

def error_response(code: int, msg: str, request_id: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> JSONResponse:
    log_msg = f"错误 {code}: {msg}"
    if request_id:
        log_msg = f"RID-{request_id}: {log_msg}"
    logger.warning(log_msg)
    return JSONResponse(
        status_code=code,
        content={"error": {"message": msg, "code": code, "type": "proxy_error"}},
        headers={**COMMON_HEADERS, **(headers or {})}
    )

def strip_potentially_harmful_html_and_normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    current_logger = logging.getLogger("EzTalkProxy.SPHASANN")
    current_logger.debug(f"Input (first 200): '{text[:200]}'")

    text_before_script_style_strip = text
    text = re.sub(r"<script[^>]*>.*?</script>|<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if text != text_before_script_style_strip:
        current_logger.debug(f"SPHASANN Step 1 (script/style strip): Applied. Text (first 200): '{text[:200]}'")
    
    text_before_html_br_p_norm = text
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)      
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)    
    text = re.sub(r"<p[^>]*>", "", text, flags=re.IGNORECASE)        
    if text != text_before_html_br_p_norm:
        current_logger.debug(f"SPHASANN Step 2 (HTML br/p norm): Applied. Text (first 200): '{text[:200]}'")

    separator_prefix_pattern_regex = r"\s*(---###)" 
    text_before_prefix_sep_processing = text
    text = re.sub(separator_prefix_pattern_regex, r"\n\n\1", text) 
    if text != text_before_prefix_sep_processing:
        current_logger.debug(f"SPHASANN Step 3 (---### prefix): Applied. Text (first 200): '{text[:200]}'")

    text_before_collapse_newlines = text
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text != text_before_collapse_newlines:
        current_logger.debug(f"SPHASANN Step 4 (collapse \\n{{3,}}): Applied. Text (first 200): '{text[:200]}'")

    lines = text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    text_after_line_stripping = "\n".join(stripped_lines)
    if text != text_after_line_stripping: 
        current_logger.debug(f"SPHASANN Step 5 (line stripping & rejoin): Applied. Text (first 200): '{text_after_line_stripping[:200]}'")
    text = text_after_line_stripping
    
    final_text = text 
    current_logger.debug(f"SPHASANN Final output (first 200): '{final_text[:200]}'")
    return final_text


def extract_sse_lines(buffer: bytearray) -> Tuple[List[bytes], bytearray]:
    lines = []
    start = 0
    while True:
        idx = buffer.find(b'\n', start)
        if idx == -1:
            break
        line = buffer[start:idx].removesuffix(b'\r')
        if len(line) > MAX_SSE_LINE_LENGTH:
            logger.warning(f"SSE行过长 ({len(line)}字节)，已跳过。")
        else:
            lines.append(line)
        start = idx + 1
    return lines, buffer[start:]

def get_current_time_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def is_gemini_2_5_model(model_name: str) -> bool:
    return "gemini-2.5" in model_name.lower()