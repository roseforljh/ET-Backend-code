import logging
import httpx
import orjson
import asyncio
import base64
import io
import os
import uuid
from typing import List

import google.generativeai as genai
import docx
from fastapi import Request, UploadFile
from fastapi.responses import StreamingResponse

from ..models.api_models import (
    ChatRequestModel,
    AppStreamEventPy,
    PartsApiMessagePy,
    AbstractApiMessagePy,
    SimpleTextApiMessagePy,
    PyTextContentPart,
    PyInlineDataContentPart,
    IncomingApiContentPart,
    PyFileUriContentPart,
    WebSearchResult
)
from ..core.config import (
    API_TIMEOUT,
    GOOGLE_API_KEY_ENV,
    MAX_DOCUMENT_UPLOAD_SIZE_MB,
    TEMP_UPLOAD_DIR,
    SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION
)
from ..utils.helpers import (
    get_current_time_iso,
    orjson_dumps_bytes_wrapper,
    extract_text_from_uploaded_document,
    to_sse_bytes,
)
from ..services.requests.facade import prepare_gemini_rest_api_request
from ..services.streaming.processor import process_openai_like_sse_stream
from ..services.streaming.error_handling import handle_stream_error
from ..services.streaming.cleanup import handle_stream_cleanup
from ..services.web_search import perform_web_search, generate_search_context_message_content

logger = logging.getLogger("EzTalkProxy.Handlers.Gemini")

IMAGE_MIME_TYPES = ["image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"]
VIDEO_MIME_TYPES = [
    "video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo", "video/x-flv",
    "video/x-matroska", "video/webm", "video/x-ms-wmv", "video/3gpp", "video/x-m4v"
]
AUDIO_MIME_TYPES = [
    "audio/wav", "audio/mpeg", "audio/aac", "audio/ogg", "audio/opus", "audio/flac", "audio/3gpp"
]

def is_google_official_api(api_address: str) -> bool:
    """Check if the API address is Google's official Gemini API"""
    if not api_address:
        return True  # Default to Google official if no address specified
    
    google_domains = [
        "generativelanguage.googleapis.com",
        "ai.google.dev",
        "googleapis.com"
    ]
    
    api_address_lower = api_address.lower()
    return any(domain in api_address_lower for domain in google_domains)


def mask_api_key(api_key: str) -> str:
    """Return a masked/fingerprinted representation of an API key for safe logging."""
    if not api_key:
        return "(empty)"
    head = api_key[:4]
    tail = api_key[-4:] if len(api_key) > 8 else "****"
    return f"{head}...{tail} (len={len(api_key)})"


def looks_like_google_api_key(api_key: str) -> bool:
    """
    Heuristic check for Google AI Studio API Key format.
    Most Google API keys start with 'AIza' when used with ?key= for REST endpoints.
    This is a heuristic; passing this check does not guarantee validity.
    """
    if not api_key:
        return False
    return api_key.startswith("AIza") and len(api_key) >= 20

async def sse_event_serializer_rest(event_data: AppStreamEventPy) -> bytes:
    # 使用统一的 SSE 序列化器（data: ␣... + 双换行）
    return to_sse_bytes(event_data)

async def handle_gemini_request(
    gemini_chat_input: ChatRequestModel,
    uploaded_files: List[UploadFile],
    fastapi_request_obj: Request,
    http_client: httpx.AsyncClient,
    request_id: str,
):
    log_prefix = f"RID-{request_id}"
    active_messages_for_llm: List[AbstractApiMessagePy] = [msg.model_copy(deep=True) for msg in gemini_chat_input.messages]
    newly_created_multimodal_parts: List[IncomingApiContentPart] = []

    # Only use user-provided API key, no fallback to environment variable
    if not gemini_chat_input.api_key:
        logger.error(f"{log_prefix}: No user-provided API key for Gemini")
        async def error_gen():
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message="No API key provided by user", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="no_api_key", timestamp=get_current_time_iso()))
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    # 先判断地址是否 Google 官方，再决定是否做 Key 形态校验与调用路径
    api_address = gemini_chat_input.api_address or ""
    is_google_official = is_google_official_api(api_address)

    # 日志打印 Key 指纹（安全掩码）
    masked = mask_api_key(gemini_chat_input.api_key)
    logger.info(f"{log_prefix}: Gemini API key fingerprint: {masked}; is_google_official={is_google_official}; base='{api_address or '(google default)'}'")

    # 仅在直连 Google 官方时，才执行 AIza 形态启发式校验
    if is_google_official and not looks_like_google_api_key(gemini_chat_input.api_key):
        logger.error(f"{log_prefix}: Provided key does not look like Google AI Studio key. key={masked}, provider='{gemini_chat_input.provider}'")
        async def error_gen():
            yield await sse_event_serializer_rest(AppStreamEventPy(
                type="error",
                message="提供的 API 密钥看起来不像 Google 官方密钥（通常以 AIza 开头）。请在设置中选择 Gemini 官方通道并填入 Google AI Studio 的 API Key，或在渠道中选择 OpenAI 兼容聚合商。",
                timestamp=get_current_time_iso()
            ))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="invalid_api_key_format", timestamp=get_current_time_iso()))
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 直连 Google 官方 → 使用 SDK 配置（用于 File API 等能力），同时 REST 仍走官方流式端点
    if is_google_official:
        genai.configure(api_key=gemini_chat_input.api_key)
        logger.info(f"{log_prefix}: Using Gemini native REST via Google official endpoint")
    else:
        # 非 Google 域名但仍为“AI Studio Build 的二次代理/2api”，按“Gemini REST 语义”使用自定义 base 直连，不再回退到 OpenAI 兼容
        logger.info(f"{log_prefix}: Using Gemini native REST via custom base (AI Studio proxy/2api): {api_address}")

    # Process uploaded files - Use comprehensive document processing like OpenAI handler
    document_texts = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            mime_type = uploaded_file.content_type.lower() if uploaded_file.content_type else ""
            filename = uploaded_file.filename or "unknown"
            
            try:
                # Check if it's a document type that should be text-extracted
                is_document_type = mime_type in SUPPORTED_DOCUMENT_MIME_TYPES_FOR_TEXT_EXTRACTION
                
                if mime_type in IMAGE_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing image for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8').replace('\n', '')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                elif is_document_type:
                    # Extract text from documents instead of sending as binary
                    logger.info(f"{log_prefix}: Extracting text from document for Gemini: {filename} ({mime_type})")
                    temp_file_path = ""
                    try:
                        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{request_id}-{uuid.uuid4()}-{filename}")
                        await uploaded_file.seek(0)
                        with open(temp_file_path, "wb") as f:
                            f.write(await uploaded_file.read())
                        
                        extracted_text = await extract_text_from_uploaded_document(
                            uploaded_file_path=temp_file_path,
                            mime_type=uploaded_file.content_type,
                            original_filename=filename
                        )
                        if extracted_text:
                            document_texts.append(extracted_text)
                            logger.info(f"{log_prefix}: Successfully extracted text from document '{filename}' for Gemini.")
                    except Exception as e:
                        logger.error(f"{log_prefix}: Failed to process document for text extraction {filename}: {e}", exc_info=True)
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                elif mime_type in AUDIO_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing audio for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    base64_data = base64.b64encode(file_bytes).decode('utf-8').replace('\n', '')
                    newly_created_multimodal_parts.append(PyInlineDataContentPart(
                        type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                    ))
                elif mime_type in VIDEO_MIME_TYPES:
                    logger.info(f"{log_prefix}: Processing video for Gemini: {filename} ({mime_type})")
                    await uploaded_file.seek(0)
                    file_bytes = await uploaded_file.read()
                    file_size = len(file_bytes)
                    
                    # Use File API for large files as recommended by Google
                    if file_size > (MAX_DOCUMENT_UPLOAD_SIZE_MB * 1024 * 1024):
                        logger.info(f"{log_prefix}: Uploading large video '{filename}' ({file_size / 1024 / 1024:.2f} MB) to Gemini File API.")
                        try:
                            # We need to run this in a separate thread as the SDK is synchronous
                            loop = asyncio.get_running_loop()
                            video_file = await loop.run_in_executor(
                                None,
                                lambda: genai.upload_file(
                                    path=io.BytesIO(file_bytes),
                                    display_name=filename,
                                    mime_type=mime_type
                                )
                            )
                            logger.info(f"{log_prefix}: Uploaded '{filename}', waiting for processing. URI: {video_file.uri}")
                            
                            # Wait for the file to be processed
                            while video_file.state.name == "PROCESSING":
                                await asyncio.sleep(5) # Non-blocking sleep
                                video_file = await loop.run_in_executor(None, lambda: genai.get_file(video_file.name))
                                logger.info(f"{log_prefix}: File '{filename}' state: {video_file.state.name}")

                            if video_file.state.name == "ACTIVE":
                                newly_created_multimodal_parts.append(PyFileUriContentPart(
                                    type="file_uri_content", fileUri=video_file.uri, mimeType=mime_type
                                ))
                                logger.info(f"{log_prefix}: File '{filename}' is active and ready to use.")
                            else:
                                logger.error(f"{log_prefix}: File '{filename}' failed to process. State: {video_file.state.name}")

                        except Exception as file_api_e:
                            logger.error(f"{log_prefix}: Gemini File API upload failed for {filename}: {file_api_e}", exc_info=True)
                    else:
                        base64_data = base64.b64encode(file_bytes).decode('utf-8').replace('\n', '')
                        newly_created_multimodal_parts.append(PyInlineDataContentPart(
                            type="inline_data_content", mimeType=mime_type, base64Data=base64_data
                        ))
                else:
                    logger.warning(f"{log_prefix}: Skipping unsupported file type for Gemini: {filename} ({mime_type})")
            except Exception as e:
                logger.error(f"{log_prefix}: Error processing file {filename} for Gemini: {e}", exc_info=True)

    # Add extracted document context to the last user message
    if document_texts:
        full_document_context = "\n\n".join(document_texts)
        full_document_context = f"--- Document Content ---\n{full_document_context}\n--- End Document ---\n\n"
        
        # Find the last user message and prepend document context
        last_user_message_idx = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            user_msg = active_messages_for_llm[last_user_message_idx]
            if isinstance(user_msg, PartsApiMessagePy):
                # Find first text part to prepend to, or insert at the beginning
                text_part_index = next((idx for idx, p in enumerate(user_msg.parts) if isinstance(p, PyTextContentPart)), -1)
                if text_part_index != -1:
                    user_msg.parts[text_part_index].text = full_document_context + user_msg.parts[text_part_index].text
                else:
                    user_msg.parts.insert(0, PyTextContentPart(type="text_content", text=full_document_context))
            elif isinstance(user_msg, SimpleTextApiMessagePy):
                # Convert to PartsApiMessagePy and prepend document context
                initial_text = full_document_context + (user_msg.content or "")
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role="user", parts=[PyTextContentPart(type="text_content", text=initial_text)]
                )
        else:
            # No user message found, create one with document context
            active_messages_for_llm.append(PartsApiMessagePy(
                role="user", parts=[PyTextContentPart(type="text_content", text=full_document_context)]
            ))

    if newly_created_multimodal_parts:
        last_user_message_idx = -1
        for i in range(len(active_messages_for_llm) - 1, -1, -1):
            if active_messages_for_llm[i].role == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx != -1:
            user_msg = active_messages_for_llm[last_user_message_idx]
            if isinstance(user_msg, PartsApiMessagePy):
                user_msg.parts.extend(newly_created_multimodal_parts)
            elif isinstance(user_msg, SimpleTextApiMessagePy):
                initial_text_part = [PyTextContentPart(type="text_content", text=user_msg.content)] if user_msg.content else []
                active_messages_for_llm[last_user_message_idx] = PartsApiMessagePy(
                    role="user", parts=initial_text_part + newly_created_multimodal_parts
                )
        else:
            active_messages_for_llm.append(PartsApiMessagePy(role="user", parts=newly_created_multimodal_parts))

    try:
        target_url, headers, json_payload = prepare_gemini_rest_api_request(
            chat_input=gemini_chat_input.model_copy(update={'messages': active_messages_for_llm}),
            request_id=request_id,
           system_prompt=gemini_chat_input.system_prompt
        )
    except Exception as prep_error:
        logger.error(f"{log_prefix}: Request preparation error: {prep_error}", exc_info=True)
        async def error_gen():
            yield await sse_event_serializer_rest(AppStreamEventPy(type="error", message=f"请求准备错误: {str(prep_error)}", timestamp=get_current_time_iso()))
            yield await sse_event_serializer_rest(AppStreamEventPy(type="finish", reason="request_error", timestamp=get_current_time_iso()))
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    async def stream_generator():
        processing_state = {}
        upstream_ok = False
        first_chunk_received = False
        full_text = ""
        original_full_text = ""  # Store original text for comparison
        grounding_supports = []
        grounding_chunks_storage = []
        
        # 401错误重试逻辑
        max_retries = 5
        retry_count = 0
        last_error_message = ""
        last_error_code = 0
        
        while retry_count < max_retries:
            try:
                async with http_client.stream("POST", target_url, headers=headers, json=json_payload, timeout=API_TIMEOUT) as response:
                    upstream_ok = response.is_success
                    if not upstream_ok:
                        error_body = await response.aread()
                        error_text = error_body.decode(errors='ignore')
                        last_error_code = response.status_code

                        # 401错误：重试
                        if response.status_code == 401:
                            retry_count += 1
                            logger.warning(f"{log_prefix}: Gemini 401 error (attempt {retry_count}/{max_retries}): {error_text[:200]}")
                            if retry_count < max_retries:
                                # 等待后重试（指数退避）
                                wait_time = min(2 ** (retry_count - 1), 10)  # 1, 2, 4, 8, 10秒
                                logger.info(f"{log_prefix}: Retrying in {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                                continue  # 重试
                            else:
                                # 5次都失败，发送错误给前端
                                last_error_message = f"Gemini API密钥验证失败 (401): 已重试{max_retries}次仍然失败。请检查您的API密钥是否正确。详细错误: {error_text[:200]}"
                                logger.error(f"{log_prefix}: {last_error_message}")
                        else:
                            # 非401错误，不重试，直接返回
                            logger.error(f"{log_prefix}: Gemini upstream error {response.status_code}: {error_text}")

                            # 检测 Cloudflare 质询（特殊处理）
                            is_cloudflare_challenge = (
                                response.status_code == 403 and 
                                ("cloudflare" in error_text.lower() or 
                                 "just a moment" in error_text.lower() or
                                 "_cf_chl_opt" in error_text or
                                 "challenge-platform" in error_text.lower())
                            )

                            # 提供友好的错误信息
                            if response.status_code == 400:
                                last_error_message = f"Gemini API请求错误 (400): 请检查模型名称和参数是否正确。详细: {error_text[:200]}"
                            elif is_cloudflare_challenge:
                                # 返回特殊错误类型，让 App 自动切换到直连模式
                                logger.warning(f"{log_prefix}: Detected Cloudflare challenge, suggest client to use direct mode")
                                last_error_message = "CLOUDFLARE_CHALLENGE_DETECTED"
                                # 发送特殊的错误事件，带上自动重试建议
                                yield await sse_event_serializer_rest(AppStreamEventPy(
                                    type="error",
                                    message=last_error_message,
                                    timestamp=get_current_time_iso(),
                                    # 添加额外的元数据供客户端识别
                                ))
                                yield await sse_event_serializer_rest(AppStreamEventPy(
                                    type="finish",
                                    reason="cloudflare_blocked_auto_retry_direct",
                                    timestamp=get_current_time_iso()
                                ))
                                return
                            elif response.status_code == 403:
                                last_error_message = f"Gemini API访问被拒绝 (403): 请检查API密钥权限或配额。详细: {error_text[:200]}"
                            elif response.status_code == 404:
                                last_error_message = f"Gemini API端点未找到 (404): 请检查模型名称是否正确。详细: {error_text[:200]}"
                            elif response.status_code == 429:
                                last_error_message = f"Gemini API请求频率过高 (429): 请稍后重试。详细: {error_text[:200]}"
                            elif response.status_code >= 500:
                                last_error_message = f"Gemini服务器内部错误 ({response.status_code}): 请稍后重试。详细: {error_text[:200]}"
                            else:
                                last_error_message = f"Gemini API错误 ({response.status_code}): {error_text[:200]}"

                        # 发送错误信息给用户
                        yield await sse_event_serializer_rest(AppStreamEventPy(
                            type="error",
                            message=last_error_message,
                            timestamp=get_current_time_iso()
                        ))
                        yield await sse_event_serializer_rest(AppStreamEventPy(
                            type="finish",
                            reason="api_error",
                            timestamp=get_current_time_iso()
                        ))
                        return

                    # 成功路径：读取上游 SSE 流
                    async for line in response.aiter_lines():
                        if not first_chunk_received:
                            first_chunk_received = True

                        if line.startswith("data:"):
                            json_str = line[len("data:"):].strip()
                            try:
                                logger.debug(f"Received from Gemini: {json_str}")
                                sse_data = orjson.loads(json_str)
                                openai_like_sse = {"id": f"gemini-{request_id}", "choices": []}

                                for candidate in sse_data.get("candidates", []):
                                    grounding_metadata = candidate.get("groundingMetadata")
                                    if grounding_metadata:
                                        search_queries = grounding_metadata.get("webSearchQueries", [])
                                        if "groundingChunks" in grounding_metadata:
                                            grounding_chunks_storage.extend(grounding_metadata["groundingChunks"])
                                        if "groundingSupports" in grounding_metadata:
                                            grounding_supports.extend(grounding_metadata["groundingSupports"])
                                        if search_queries:
                                            logger.info(f"{log_prefix}: Gemini used web search with queries: {search_queries}")
                                        if grounding_chunks_storage:
                                            web_results = [
                                                WebSearchResult(
                                                    title=chunk.get("web", {}).get("title", "Unknown Source"),
                                                    url=chunk.get("web", {}).get("uri", "#"),
                                                    snippet=f"Source: {chunk.get('web', {}).get('title', 'N/A')}"
                                                )
                                                for chunk in grounding_chunks_storage if chunk.get("web")
                                            ]
                                            if web_results:
                                                yield await sse_event_serializer_rest(AppStreamEventPy(
                                                    type="web_search_results",
                                                    web_search_results=web_results
                                                ))

                                    content_parts = candidate.get("content", {}).get("parts", [])
                                    # 可能一次 data 帧里包含多段 part（thought/text/inlineData 混合）
                                    # 之前仅保留了最后一个 text；这里改为聚合同一帧中的所有文本。
                                    has_any_thought = False
                                    aggregated_thoughts: list[str] = []
                                    aggregated_texts: list[str] = []

                                    for part in content_parts:
                                        # 思考片段
                                        if "thought" in part and part.get("text"):
                                            has_any_thought = True
                                            aggregated_thoughts.append(part["text"])
                                        # 代码执行片段
                                        elif "executableCode" in part:
                                            code_data = part.get("executableCode", {})
                                            code_text = code_data.get("code", "")
                                            code_lang = code_data.get("language", "python")
                                            if code_text:
                                                logger.info(f"{log_prefix}: Received executable code ({code_lang}), length={len(code_text)}")
                                                yield await sse_event_serializer_rest(AppStreamEventPy(
                                                    type="code_executable",
                                                    executable_code=code_text,
                                                    code_language=code_lang
                                                ))
                                        # 代码执行结果
                                        elif "codeExecutionResult" in part:
                                            result_data = part.get("codeExecutionResult", {})
                                            output_text = result_data.get("output", "")
                                            outcome = result_data.get("outcome", "OUTCOME_OK")
                                            outcome_status = "success" if outcome == "OUTCOME_OK" else "error"
                                            logger.info(f"{log_prefix}: Received code execution result, outcome={outcome}, output_length={len(output_text)}")
                                            yield await sse_event_serializer_rest(AppStreamEventPy(
                                                type="code_execution_result",
                                                code_execution_output=output_text,
                                                code_execution_outcome=outcome_status
                                            ))
                                        # 图片内联
                                        elif "inlineData" in part:
                                            mime_type = part["inlineData"].get("mimeType")
                                            base64_data = part["inlineData"].get("data")
                                            if mime_type and base64_data:
                                                image_url = f"data:{mime_type};base64,{base64_data}"
                                                if mime_type.startswith("image/"):
                                                    logger.info(f"{log_prefix}: Received inline image from code execution, mime={mime_type}")
                                                    yield await sse_event_serializer_rest(AppStreamEventPy(
                                                        type="code_execution_result",
                                                        image_url=image_url,
                                                        code_execution_outcome="success"
                                                    ))
                                                else:
                                                    # 图像生成模式
                                                    yield await sse_event_serializer_rest(AppStreamEventPy(
                                                        type="image_generation",
                                                        image_url=image_url
                                                    ))
                                                    yield await sse_event_serializer_rest(AppStreamEventPy(
                                                        type="finish",
                                                        reason="image_generated",
                                                        timestamp=get_current_time_iso()
                                                    ))
                                                    return
                                        # 普通文本
                                        elif "text" in part:
                                            text_chunk = part["text"] or ""
                                            if text_chunk:
                                                aggregated_texts.append(text_chunk)

                                    # 先发思考聚合（若有）
                                    if has_any_thought and aggregated_thoughts:
                                        reasoning_text = "".join(aggregated_thoughts)
                                        yield await sse_event_serializer_rest(AppStreamEventPy(type="reasoning", text=reasoning_text))

                                    # 再发文本聚合（若有）
                                    if aggregated_texts:
                                        text_payload = "".join(aggregated_texts)
                                        logger.debug(f"{log_prefix}: Aggregated text in one frame, length={len(text_payload)}")
                                        logger.debug(f"{log_prefix}: Text payload preview: {repr(text_payload[:100])}")
                                        # 保留“原样输出，不做格式修复”
                                        original_full_text += text_payload
                                        full_text += text_payload

                                        choice = {
                                            "delta": {"content": text_payload},
                                            "finish_reason": candidate.get("finishReason")
                                        }
                                        openai_like_sse = {"id": f"gemini-{request_id}", "choices": [choice]}
                                        async for event in process_openai_like_sse_stream(openai_like_sse, processing_state, request_id):
                                            yield await sse_event_serializer_rest(AppStreamEventPy(**event))
                                            await asyncio.sleep(0)
                            except orjson.JSONDecodeError:
                                logger.warning(f"{log_prefix}: Skipping non-JSON line in Gemini stream: {line}")

                # 成功读完流，跳出重试循环
                break
            except Exception as e:
                logger.error(f"{log_prefix}: An error occurred during the Gemini stream (attempt {retry_count+1}/{max_retries}): {e}", exc_info=True)
                async for error_event in handle_stream_error(e, request_id, upstream_ok, first_chunk_received):
                    yield error_event
                return
        # 记录最终输出统计信息，但不重复发送内容
        if original_full_text:
            logger.info(f"{log_prefix}: Final output ready")
            logger.info(f"{log_prefix}: Original full text length: {len(original_full_text)}")
            logger.info(f"{log_prefix}: Streamed full text length: {len(full_text)}")
            logger.info(f"{log_prefix}: Content already streamed, skipping duplicate final send")

        is_native_thinking = bool(gemini_chat_input.generation_config and gemini_chat_input.generation_config.thinking_config)
        use_custom_sep = False

        # 移除引用处理逻辑，直接传输原生内容
        if grounding_supports and grounding_chunks_storage:
            logger.info(f"Raw grounding data available. Supports: {len(grounding_supports)}, Chunks: {len(grounding_chunks_storage)}")
            logger.info(f"Raw text without citation processing: {original_full_text[:200]}...")

        async for final_event in handle_stream_cleanup(processing_state, request_id, upstream_ok, use_custom_sep, gemini_chat_input.provider):
            yield final_event

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )