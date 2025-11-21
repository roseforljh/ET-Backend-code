import  logging
import  re
import  orjson
from  typing  import  Dict,  Any,  AsyncGenerator

from  ...models.api_models  import  AppStreamEventPy
from  ...utils.helpers  import  get_current_time_iso
from  .flush  import  MarkdownBlockDetector  as  _FlushDetector
# format_fixer removed per requirement

logger  =  logging.getLogger("EzTalkProxy.StreamProcessors")

def get_strategy_for_text(text: str) -> tuple[Any, str]:
    """
    返回无操作策略，保持AI输出原样。
    Always returns (None, "general").
    """
    return None, "general"
 
def lightweight_cleanup(text: str, is_streaming: bool = True) -> str:
    """
    最小化清理：仅统一换行符，保持AI原始输出。
    
    参数：
        text: 待清理的文本
        is_streaming: 是否为流式输出模式（默认True）
    
    清理内容：
    - 统一换行符格式（\r\n -> \n）
    """
    if not text:
        return text
    
    # 仅统一换行符格式，其余保持原样
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    return text

# --- Streaming-safe helpers for code fences ---
_fence_line_re = re.compile(r'^(\s*)([`~]{3,})(.*)$')

def selective_cleanup_with_code_fence(text: str, state: Dict[str, Any]) -> str:
    """
    直接返回原始文本，不做任何清理。
    保持函数签名以兼容现有代码。
    """
    return text

def _is_meaningful(chunk: str) -> bool:
    """
    判定增量是否有实际内容：过滤只包含空白字符（空格/制表符/换行）的块。
    """
    return bool(chunk) and bool(chunk.strip())

def _ensure_code_blocks_closed(text: str) -> str:
    """
    直接返回原始文本，不做代码块闭合修复。
    保持函数签名以兼容现有代码。
    """
    return text

def _try_parse_openai_gemini_code_events(content_chunk: Any) -> tuple[bool, list[Dict[str, Any]]]:
    """
    解析 OpenAI 兼容流中的 Gemini 代码执行特殊增量。
    - content 可能是 JSON 字符串，包含 executableCode 或 codeExecutionResult
    - 返回: (是否已处理, 事件列表[dict])
    """
    events: list[Dict[str, Any]] = []
    try:
        # 统一转文本
        if isinstance(content_chunk, (bytes, bytearray)):
            text = content_chunk.decode("utf-8", "ignore")
        else:
            text = str(content_chunk)
        s = (text or "").strip()
        # 必须形如 JSON
        if not s or not (s.startswith("{") or s.startswith("[")):
            return False, []
        data = orjson.loads(s)

        def handle_obj(obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            # 可执行代码
            if "executableCode" in obj:
                ec = obj.get("executableCode") or {}
                code = ec.get("code") or ""
                lang = ec.get("language") or "python"
                if code:
                    events.append({
                        "type": "code_executable",
                        "executable_code": code,
                        "code_language": lang
                    })
            # 执行结果
            if "codeExecutionResult" in obj:
                res = obj.get("codeExecutionResult") or {}
                output = res.get("output") or ""
                outcome = res.get("outcome") or res.get("status") or ""
                if str(outcome).upper() in ("OUTCOME_OK", "SUCCESS", "OK"):
                    outcome_norm = "success"
                elif outcome:
                    outcome_norm = "error"
                else:
                    outcome_norm = "success" if output else None
                ev: Dict[str, Any] = {
                    "type": "code_execution_result",
                    "code_execution_output": output
                }
                if outcome_norm:
                    ev["code_execution_outcome"] = outcome_norm
                events.append(ev)

        if isinstance(data, dict):
            handle_obj(data)
        elif isinstance(data, list):
            for item in data:
                handle_obj(item)

        return (len(events) > 0), events
    except Exception:
        # 任意解析失败视为非代码事件
        return False, []

async def process_openai_like_sse_stream(
    parsed_sse_data: Dict[str, Any],
    current_processing_state: Dict[str, Any],
    request_id: str,
    request_data = None
) -> AsyncGenerator[Dict[str, Any], None]:
    log_prefix = f"RID-{request_id}"
    
    state = current_processing_state
    state.setdefault("had_any_reasoning", False)
    state.setdefault("reasoning_finish_event_sent", False)
    # 分离正文与思考的累积，避免最终清理把思考混入正文
    state.setdefault("accumulated_content", "")
    state.setdefault("accumulated_reasoning", "")
    state.setdefault("detected_type", "general")  # 存储检测到的内容类型
    # Track fenced code block state across streaming chunks
    state.setdefault("code_fence", {"in": False, "marker": ""})

    for choice in parsed_sse_data.get('choices', []):
        delta = choice.get('delta', {})
        content_chunk = delta.get("content")
        reasoning_chunk = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking") or delta.get("thoughts")
        finish_reason = choice.get("finish_reason")

        if reasoning_chunk:
            # 🔍 [STREAM_DEBUG] 记录reasoning事件发送
            logger.info(f"[STREAM_DEBUG] {log_prefix} ✅ SENDING reasoning event: len={len(reasoning_chunk)}")
            yield {"type": "reasoning", "text": str(reasoning_chunk), "timestamp": get_current_time_iso()}
            state["had_any_reasoning"] = True
            # 分开累积思考文本，绝不混入正文最终清理
            state["accumulated_reasoning"] = str(state.get("accumulated_reasoning", "")) + str(reasoning_chunk)

        # 先尝试解析 Gemini 的代码执行特殊事件（OpenAI 兼容流）
        if _is_meaningful(content_chunk):
            handled_code, code_events = _try_parse_openai_gemini_code_events(content_chunk)
            if handled_code and code_events:
                for ev in code_events:
                    ev["timestamp"] = get_current_time_iso()
                    logger.info(f"[STREAM_DEBUG] {log_prefix} ✅ SENDING code event: type={ev.get('type')}, len_output={len(ev.get('code_execution_output','') or ev.get('executable_code',''))}")
                    yield ev
                # 解析到代码事件则不按普通文本重复发送
                content_chunk = None

        # 仅对正文进行清理与累积，用于类型检测与刷新判定
        if _is_meaningful(content_chunk):
            # 🎯 停用后端预处理：直接透传原始增量
            cleaned_for_accumulation = str(content_chunk)
            if _is_meaningful(cleaned_for_accumulation):
                state["accumulated_content"] += cleaned_for_accumulation

            # 统一策略：立即发送所有内容
            logger.info(f"[STREAM_DEBUG] {log_prefix} Immediate flush, chunk_len={len(cleaned_for_accumulation)}")

            if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                # 如果我们已经处理了 reasoning，并且现在有了实际内容，那么 reasoning 阶段结束
                if content_chunk:
                    logger.info(f"[STREAM_DEBUG] {log_prefix} Sending reasoning_finish event")
                    yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                    state["reasoning_finish_event_sent"] = True

            # 动态检测 output_type 并存储（仅基于正文）
            _, detected_type = get_strategy_for_text(state["accumulated_content"])
            state["detected_type"] = detected_type

            # 发送正文增量（已清理过，直接发送，避免二次清理）
            if content_chunk and _is_meaningful(cleaned_for_accumulation):
                # 🔍 [STREAM_DEBUG] 记录content事件发送
                logger.info(f"[STREAM_DEBUG] {log_prefix} ✅ SENDING content event: len={len(cleaned_for_accumulation)}, type={detected_type}, preview='{cleaned_for_accumulation[:50]}'")
                yield {
                    "type": "content",
                    "text": cleaned_for_accumulation,  # 🎯 修复：使用已清理的内容，避免重复调用lightweight_cleanup
                    "output_type": detected_type,
                    "timestamp": get_current_time_iso()
                }

        if finish_reason:
            if state["had_any_reasoning"] and not state["reasoning_finish_event_sent"]:
                yield {"type": "reasoning_finish", "timestamp": get_current_time_iso()}
                state["reasoning_finish_event_sent"] = True

            # 停用诊断性最终修复：仅记录长度信息，直接结束
            accumulated = state.get("accumulated_content", "")
            if accumulated:
                logger.info(f"{log_prefix} Stream ending with accumulated content length: {len(accumulated)} chars")
            
            yield {"type": "finish", "reason": finish_reason, "timestamp": get_current_time_iso()}