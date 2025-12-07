"""
预测性 TTS 处理器

实现 LLM 输出与 TTS 合成的并行处理，减少语音响应延迟。

核心思想：
- LLM 输出片段后立即提交 TTS 任务
- TTS 任务在后台并行执行
- 按顺序输出音频，确保播放连贯性
- 支持 TTS 预热，减少首次请求延迟

优化特性：
- 首句优先处理：第一个任务使用更高优先级
- TTS 预热：预建立连接，减少冷启动延迟
- 智能重试：支持指数退避的错误重试
- 流式输出模式：支持边生成边发送（适用于阿里云等平台）
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Awaitable, Tuple
from enum import Enum

logger = logging.getLogger("EzTalkProxy.Services.PredictiveTTS")


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TTSTask:
    """TTS 任务数据结构"""
    sequence_id: int              # 序列号，确保顺序
    text: str                     # 待合成文本
    status: TaskStatus = TaskStatus.PENDING
    audio_chunks: List[bytes] = field(default_factory=list)
    retry_count: int = 0          # 已重试次数
    error: Optional[str] = None   # 错误信息
    completed_event: asyncio.Event = field(default_factory=asyncio.Event)


class PredictiveTTSProcessor:
    """
    预测性 TTS 处理器
    
    特点：
    - 并发执行多个 TTS 请求（max_concurrent=5）
    - 失败自动重试（max_retry=2）
    - 按顺序输出音频
    - 支持 Gemini、Minimax、SiliconFlow、OpenAI、Aliyun 五种平台
    - 首句优先处理：第一个任务立即执行，不受并发限制
    - TTS 预热：可预建立连接减少冷启动延迟
    """
    
    def __init__(
        self,
        tts_executor: Callable[[str], Awaitable[AsyncGenerator[bytes, None]]],
        max_concurrent: int = 5,
        max_retry: int = 2,
        task_timeout: float = 30.0,
        first_task_timeout: float = 15.0,  # 首句使用更短超时，快速失败
        enable_warmup: bool = True,         # 是否启用预热
    ):
        """
        初始化预测性 TTS 处理器
        
        Args:
            tts_executor: TTS 执行函数，接收文本，返回音频流
            max_concurrent: 最大并发 TTS 请求数
            max_retry: 最大重试次数
            task_timeout: 单任务超时时间（秒）
            first_task_timeout: 首句任务超时时间（秒），使用更短超时快速失败
            enable_warmup: 是否启用 TTS 预热
        """
        self.tts_executor = tts_executor
        self.max_concurrent = max_concurrent
        self.max_retry = max_retry
        self.task_timeout = task_timeout
        self.first_task_timeout = first_task_timeout
        self.enable_warmup = enable_warmup
        
        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 任务存储：sequence_id -> TTSTask
        self.tasks: Dict[int, TTSTask] = {}
        
        # 下一个应输出的序列号
        self.next_output_id = 0
        
        # 输入是否已完成
        self.input_complete = False
        
        # 总任务数（用于判断是否全部完成）
        self.total_tasks = 0
        
        # 用于通知输出线程有新任务完成
        self.output_notify = asyncio.Event()
        
        # 活跃的任务 Future
        self._active_tasks: List[asyncio.Task] = []
        
        # 预热状态
        self._warmup_complete = False
        self._warmup_task: Optional[asyncio.Task] = None
        
        # 性能统计
        self._first_task_start_time: Optional[float] = None
        self._first_audio_time: Optional[float] = None
        
        logger.info(f"PredictiveTTSProcessor initialized: max_concurrent={max_concurrent}, "
                   f"max_retry={max_retry}, first_timeout={first_task_timeout}, warmup={enable_warmup}")
    
    async def warmup(self) -> bool:
        """
        预热 TTS 连接，减少首次请求延迟
        
        通过发送一个极短的文本来预建立连接和初始化 TTS 引擎。
        
        Returns:
            bool: 预热是否成功
        """
        if not self.enable_warmup or self._warmup_complete:
            return True
        
        warmup_text = "。"  # 最短可发音文本
        start_time = time.time()
        
        try:
            logger.info("[TTS Warmup] 开始预热...")
            async for _ in self.tts_executor(warmup_text):
                # 只需要触发连接，不需要收集音频
                break
            
            elapsed = (time.time() - start_time) * 1000
            self._warmup_complete = True
            logger.info(f"[TTS Warmup] ✓ 预热完成: {elapsed:.0f}ms")
            return True
            
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.warning(f"[TTS Warmup] ⚠ 预热失败 ({elapsed:.0f}ms): {e}")
            # 预热失败不阻塞后续流程
            self._warmup_complete = True
            return False
    
    async def start_warmup_async(self) -> None:
        """
        异步启动预热（不阻塞调用者）
        
        在 Chat 开始前调用，让预热与其他初始化并行执行。
        """
        if not self.enable_warmup or self._warmup_complete:
            return
        
        self._warmup_task = asyncio.create_task(self.warmup())
    
    async def wait_warmup(self) -> bool:
        """
        等待预热完成
        
        Returns:
            bool: 预热是否成功
        """
        if self._warmup_task:
            return await self._warmup_task
        return self._warmup_complete
    
    async def submit_task(self, sequence_id: int, text: str) -> None:
        """
        提交 TTS 任务
        
        Args:
            sequence_id: 任务序列号
            text: 待合成文本
        """
        if not text.strip():
            logger.debug(f"Skipping empty text for sequence {sequence_id}")
            return
        
        # 首个任务记录开始时间
        if sequence_id == 0:
            self._first_task_start_time = time.time()
            
        task = TTSTask(sequence_id=sequence_id, text=text)
        self.tasks[sequence_id] = task
        self.total_tasks = max(self.total_tasks, sequence_id + 1)
        
        # 记录详细的任务信息
        is_first = sequence_id == 0
        logger.info(f"[TTS Task {sequence_id}] {'[首句] ' if is_first else ''}提交任务: "
                   f"'{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # 首句任务：立即执行，不受信号量限制（优先处理）
        if is_first:
            async_task = asyncio.create_task(self._process_first_task(task))
        else:
            async_task = asyncio.create_task(self._process_task_with_retry(task))
        
        self._active_tasks.append(async_task)
    
    async def _process_first_task(self, task: TTSTask) -> None:
        """
        处理首句任务（优先级最高，不受信号量限制）
        
        特点：
        - 立即执行，不等待信号量
        - 使用更短的超时时间
        - 记录首字延迟统计
        
        Args:
            task: TTS 任务
        """
        task.status = TaskStatus.PROCESSING
        start_time = time.time()
        
        for attempt in range(self.max_retry + 1):
            try:
                logger.info(f"[TTS Task 0] [首句] 开始处理 (尝试 {attempt + 1}/{self.max_retry + 1})")
                await self._execute_tts(task, timeout=self.first_task_timeout)
                task.status = TaskStatus.COMPLETED
                
                # 计算音频数据总大小和延迟
                total_bytes = sum(len(chunk) for chunk in task.audio_chunks)
                elapsed = (time.time() - start_time) * 1000
                
                # 记录首句音频时间
                if self._first_audio_time is None and task.audio_chunks:
                    self._first_audio_time = time.time()
                    if self._first_task_start_time:
                        first_audio_latency = (self._first_audio_time - self._first_task_start_time) * 1000
                        logger.info(f"[TTS Task 0] [首句] ✓ 完成: {len(task.audio_chunks)} 块, "
                                   f"{total_bytes} 字节, 首字延迟={first_audio_latency:.0f}ms")
                else:
                    logger.info(f"[TTS Task 0] [首句] ✓ 完成: {len(task.audio_chunks)} 块, "
                               f"{total_bytes} 字节, 处理耗时={elapsed:.0f}ms")
                break
                
            except asyncio.TimeoutError:
                task.error = "Timeout"
                elapsed = (time.time() - start_time) * 1000
                logger.warning(f"[TTS Task 0] [首句] ⚠ 超时 ({elapsed:.0f}ms) (尝试 {attempt + 1}/{self.max_retry + 1})")
                if attempt < self.max_retry:
                    task.retry_count += 1
                    await asyncio.sleep(0.3)  # 首句使用更短的重试间隔
                else:
                    task.status = TaskStatus.FAILED
                    logger.error(f"[TTS Task 0] [首句] ✗ 失败: 重试 {self.max_retry + 1} 次后仍超时")
                    
            except Exception as e:
                task.error = str(e)
                elapsed = (time.time() - start_time) * 1000
                logger.warning(f"[TTS Task 0] [首句] ⚠ 错误 ({elapsed:.0f}ms): {e} "
                              f"(尝试 {attempt + 1}/{self.max_retry + 1})")
                
                if attempt < self.max_retry:
                    task.retry_count += 1
                    await asyncio.sleep(0.3)
                else:
                    task.status = TaskStatus.FAILED
                    logger.error(f"[TTS Task 0] [首句] ✗ 失败: {e}")
        
        # 标记完成（无论成功或失败）
        task.completed_event.set()
        self.output_notify.set()
    
    async def _process_task_with_retry(self, task: TTSTask) -> None:
        """
        处理单个 TTS 任务，支持重试
        
        Args:
            task: TTS 任务
        """
        async with self.semaphore:  # 并发控制
            task.status = TaskStatus.PROCESSING
            
            for attempt in range(self.max_retry + 1):
                try:
                    logger.info(f"[TTS Task {task.sequence_id}] 开始处理 (尝试 {attempt + 1}/{self.max_retry + 1})")
                    await self._execute_tts(task)
                    task.status = TaskStatus.COMPLETED
                    
                    # 计算音频数据总大小
                    total_bytes = sum(len(chunk) for chunk in task.audio_chunks)
                    logger.info(f"[TTS Task {task.sequence_id}] ✓ 完成: {len(task.audio_chunks)} 块, {total_bytes} 字节")
                    break
                    
                except asyncio.TimeoutError:
                    task.error = "Timeout"
                    logger.warning(f"[TTS Task {task.sequence_id}] ⚠ 超时 (尝试 {attempt + 1}/{self.max_retry + 1})")
                    if attempt < self.max_retry:
                        task.retry_count += 1
                        await asyncio.sleep(0.5)  # 短暂等待后重试
                    else:
                        task.status = TaskStatus.FAILED
                        logger.error(f"[TTS Task {task.sequence_id}] ✗ 失败: 重试 {self.max_retry + 1} 次后仍超时")
                        
                except Exception as e:
                    task.error = str(e)
                    error_str = str(e).lower()
                    
                    # 检查是否是限速错误 (429)
                    is_rate_limit = "429" in error_str or "rate limit" in error_str or "too many" in error_str
                    
                    logger.warning(f"[TTS Task {task.sequence_id}] ⚠ 错误: {e} (尝试 {attempt + 1}/{self.max_retry + 1})")
                    
                    if attempt < self.max_retry:
                        task.retry_count += 1
                        
                        # 如果是限速错误，使用指数退避
                        if is_rate_limit:
                            backoff_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s...
                            logger.info(f"[TTS Task {task.sequence_id}] 检测到限速，等待 {backoff_time:.1f}s 后重试")
                            await asyncio.sleep(backoff_time)
                        else:
                            await asyncio.sleep(0.5)
                    else:
                        task.status = TaskStatus.FAILED
                        logger.error(f"[TTS Task {task.sequence_id}] ✗ 失败: {e}")
            
            # 标记完成（无论成功或失败）
            task.completed_event.set()
            self.output_notify.set()
    
    async def _execute_tts(self, task: TTSTask, timeout: float = None) -> None:
        """
        执行 TTS 合成
        
        Args:
            task: TTS 任务
            timeout: 超时时间（秒），默认使用 self.task_timeout
        """
        task.audio_chunks = []
        actual_timeout = timeout or self.task_timeout
        
        # 使用超时包装
        async def collect_audio():
            async for chunk in self.tts_executor(task.text):
                if chunk:
                    task.audio_chunks.append(chunk)
        
        await asyncio.wait_for(collect_audio(), timeout=actual_timeout)
    
    def mark_input_complete(self) -> None:
        """标记输入已完成，不会再有新任务提交"""
        self.input_complete = True
        self.output_notify.set()
        logger.debug("Input marked as complete")
    
    async def yield_audio_in_order(self) -> AsyncGenerator[bytes, None]:
        """
        按顺序输出音频
        
        Yields:
            音频数据块
        """
        while True:
            # 检查下一个应输出的任务
            if self.next_output_id in self.tasks:
                task = self.tasks[self.next_output_id]
                
                # 等待任务完成
                await task.completed_event.wait()
                
                # 输出音频
                if task.status == TaskStatus.COMPLETED and task.audio_chunks:
                    total_bytes = sum(len(chunk) for chunk in task.audio_chunks)
                    for chunk in task.audio_chunks:
                        yield chunk
                    logger.info(f"[TTS Output {self.next_output_id}] 输出音频: {len(task.audio_chunks)} 块, {total_bytes} 字节")
                elif task.status == TaskStatus.FAILED:
                    logger.error(f"[TTS Output {self.next_output_id}] ✗ 跳过失败任务: {task.error}")
                elif not task.audio_chunks:
                    logger.warning(f"[TTS Output {self.next_output_id}] ⚠ 任务完成但无音频数据")
                
                # 清理已输出的任务
                del self.tasks[self.next_output_id]
                self.next_output_id += 1
                
            else:
                # 检查是否所有任务都已完成
                if self.input_complete and self.next_output_id >= self.total_tasks:
                    logger.debug("All tasks completed, stopping output")
                    break
                
                # 等待新任务完成
                self.output_notify.clear()
                try:
                    await asyncio.wait_for(self.output_notify.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 超时后继续检查
                    pass
    
    async def cleanup(self) -> None:
        """清理资源，取消所有未完成的任务"""
        for async_task in self._active_tasks:
            if not async_task.done():
                async_task.cancel()
        
        # 等待所有任务结束
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        
        self._active_tasks.clear()
        self.tasks.clear()
        logger.debug("PredictiveTTSProcessor cleaned up")


class StreamingTTSProcessor:
    """
    流式 TTS 处理器
    
    与 PredictiveTTSProcessor 的区别：
    - 边生成边发送：每个音频块生成时立即通过回调发送
    - 适用于阿里云等 TTS 延迟较高的平台
    - 避免等待整个任务完成才输出
    
    工作流程：
    1. 提交任务时立即开始处理
    2. TTS 每生成一个音频块，立即通过 on_audio_chunk 回调发送
    3. 使用队列保证顺序：只有当前任务完成后才处理下一个
    """
    
    def __init__(
        self,
        tts_executor: Callable[[str], Awaitable[AsyncGenerator[bytes, None]]],
        on_audio_chunk: Callable[[bytes], Awaitable[None]],
        max_retry: int = 3,
        task_timeout: float = 60.0,
    ):
        """
        初始化流式 TTS 处理器
        
        Args:
            tts_executor: TTS 执行函数，接收文本，返回音频流
            on_audio_chunk: 音频块回调函数，每生成一个块立即调用
            max_retry: 最大重试次数
            task_timeout: 单任务超时时间（秒）
        """
        self.tts_executor = tts_executor
        self.on_audio_chunk = on_audio_chunk
        self.max_retry = max_retry
        self.task_timeout = task_timeout
        
        # 任务队列：(sequence_id, text)
        self._task_queue: asyncio.Queue[Optional[Tuple[int, str]]] = asyncio.Queue()
        
        # 处理协程
        self._processor_task: Optional[asyncio.Task] = None
        
        # 状态
        self._is_running = False
        self._input_complete = False
        
        # 性能统计
        self._first_chunk_time: Optional[float] = None
        self._start_time: Optional[float] = None
        
        logger.info(f"StreamingTTSProcessor initialized: max_retry={max_retry}, timeout={task_timeout}")
    
    async def start(self) -> None:
        """启动处理器"""
        if self._is_running:
            return
        
        self._is_running = True
        self._start_time = time.time()
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.debug("StreamingTTSProcessor started")
    
    async def submit_task(self, sequence_id: int, text: str) -> None:
        """
        提交 TTS 任务
        
        Args:
            sequence_id: 任务序列号
            text: 待合成文本
        """
        if not text.strip():
            logger.debug(f"Skipping empty text for sequence {sequence_id}")
            return
        
        await self._task_queue.put((sequence_id, text))
        logger.info(f"[Streaming TTS {sequence_id}] 提交任务: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    def mark_input_complete(self) -> None:
        """标记输入已完成"""
        self._input_complete = True
        # 发送结束信号
        asyncio.create_task(self._task_queue.put(None))
        logger.debug("StreamingTTSProcessor input marked complete")
    
    async def _process_loop(self) -> None:
        """任务处理循环"""
        while self._is_running:
            try:
                # 等待下一个任务
                task = await self._task_queue.get()
                
                # 结束信号
                if task is None:
                    logger.debug("StreamingTTSProcessor received end signal")
                    break
                
                sequence_id, text = task
                await self._process_single_task(sequence_id, text)
                
            except asyncio.CancelledError:
                logger.debug("StreamingTTSProcessor cancelled")
                break
            except Exception as e:
                logger.error(f"StreamingTTSProcessor loop error: {e}")
        
        self._is_running = False
        logger.debug("StreamingTTSProcessor loop ended")
    
    async def _process_single_task(self, sequence_id: int, text: str) -> None:
        """
        处理单个 TTS 任务，边生成边发送
        
        Args:
            sequence_id: 任务序列号
            text: 待合成文本
        """
        start_time = time.time()
        is_first_task = sequence_id == 0
        
        for attempt in range(self.max_retry + 1):
            try:
                logger.info(f"[Streaming TTS {sequence_id}] {'[首句] ' if is_first_task else ''}开始处理 "
                           f"(尝试 {attempt + 1}/{self.max_retry + 1})")
                
                chunk_count = 0
                total_bytes = 0
                first_chunk_sent = False
                
                async def process_with_timeout():
                    nonlocal chunk_count, total_bytes, first_chunk_sent
                    
                    async for chunk in self.tts_executor(text):
                        if chunk:
                            chunk_count += 1
                            total_bytes += len(chunk)
                            
                            # 记录首个音频块时间
                            if not first_chunk_sent:
                                first_chunk_sent = True
                                if is_first_task and self._first_chunk_time is None:
                                    self._first_chunk_time = time.time()
                                    if self._start_time:
                                        latency = (self._first_chunk_time - self._start_time) * 1000
                                        logger.info(f"[Streaming TTS {sequence_id}] [首句] 首音频块延迟: {latency:.0f}ms")
                            
                            # 立即发送音频块
                            await self.on_audio_chunk(chunk)
                
                await asyncio.wait_for(process_with_timeout(), timeout=self.task_timeout)
                
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"[Streaming TTS {sequence_id}] ✓ 完成: {chunk_count} 块, "
                           f"{total_bytes} 字节, 耗时={elapsed:.0f}ms")
                return  # 成功，退出重试循环
                
            except asyncio.TimeoutError:
                elapsed = (time.time() - start_time) * 1000
                logger.warning(f"[Streaming TTS {sequence_id}] ⚠ 超时 ({elapsed:.0f}ms) "
                              f"(尝试 {attempt + 1}/{self.max_retry + 1})")
                if attempt < self.max_retry:
                    await asyncio.sleep(0.5)
                else:
                    logger.error(f"[Streaming TTS {sequence_id}] ✗ 失败: 重试后仍超时")
                    
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "rate limit" in error_str
                
                logger.warning(f"[Streaming TTS {sequence_id}] ⚠ 错误 ({elapsed:.0f}ms): {e} "
                              f"(尝试 {attempt + 1}/{self.max_retry + 1})")
                
                if attempt < self.max_retry:
                    if is_rate_limit:
                        backoff = (2 ** attempt) * 1.0
                        logger.info(f"[Streaming TTS {sequence_id}] 限速，等待 {backoff:.1f}s")
                        await asyncio.sleep(backoff)
                    else:
                        await asyncio.sleep(0.5)
                else:
                    logger.error(f"[Streaming TTS {sequence_id}] ✗ 失败: {e}")
    
    async def wait_complete(self) -> None:
        """等待所有任务完成"""
        if self._processor_task:
            await self._processor_task
    
    async def cleanup(self) -> None:
        """清理资源"""
        self._is_running = False
        
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.debug("StreamingTTSProcessor cleaned up")


class PredictiveTTSManager:
    """
    预测性 TTS 管理器
    
    封装 TTS 执行逻辑，支持多平台
    """
    
    def __init__(self, tts_config: Dict[str, Any]):
        """
        初始化管理器
        
        Args:
            tts_config: TTS 配置，包含 platform, api_key, api_url, model, voice_name
        """
        self.tts_config = tts_config
        self.platform = tts_config.get("platform", "Gemini")
        
        # 延迟导入，避免循环依赖
        from .voice import google_handler, openai_handler, minimax_handler, siliconflow_handler
        from .voice import aliyun_tts_handler
        from .voice.validator import validate_voice_config
        
        self.google_handler = google_handler
        self.openai_handler = openai_handler
        self.minimax_handler = minimax_handler
        self.siliconflow_handler = siliconflow_handler
        self.aliyun_tts_handler = aliyun_tts_handler
        self.validate_voice_config = validate_voice_config
    
    async def execute_tts(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        执行 TTS 合成
        
        Args:
            text: 待合成文本
            
        Yields:
            音频数据块
        """
        platform = self.tts_config.get("platform", "Gemini")
        voice_name = self.tts_config.get("voice_name", "Kore")
        
        # 校验配置
        self.validate_voice_config(platform, voice_name)
        
        try:
            if platform == "Minimax":
                async for chunk in self.minimax_handler.synthesize_minimax_t2a_stream(
                    text=text,
                    voice_id=voice_name,
                    api_url=self.tts_config["api_url"],
                    api_key=self.tts_config["api_key"]
                ):
                    yield chunk
                    
            elif platform == "SiliconFlow":
                # 从 voice_streaming.py 的 AUDIO_FORMAT_CONFIG 获取配置
                # SiliconFlow: pcm, 32000 (Opus 格式存在解码兼容性问题，改用 PCM)
                async for chunk in self.siliconflow_handler.process_tts_stream(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    api_url=self.tts_config["api_url"],
                    model=self.tts_config["model"],
                    voice=voice_name,
                    response_format="pcm",
                    sample_rate=32000
                ):
                    yield chunk
                    
            elif platform == "OpenAI":
                # OpenAI TTS 不是真正的流式，返回完整文件
                wav_data = await self.openai_handler.process_tts(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    voice_name=voice_name,
                    model=self.tts_config["model"],
                    api_url=self.tts_config.get("api_url")
                )
                if wav_data:
                    yield wav_data
                else:
                    raise ValueError("OpenAI TTS returned empty data")
                    
            elif platform == "Aliyun":
                # 阿里云 Qwen-TTS HTTP API 流式合成（更稳定）
                async for chunk in self.aliyun_tts_handler.process_tts_stream(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    api_url=self.tts_config.get("api_url"),
                    model=self.tts_config.get("model", "qwen3-tts-flash"),
                    voice=voice_name,
                    sample_rate=24000
                ):
                    yield chunk
                    
            else:  # Gemini
                pcm_data = self.google_handler.process_tts(
                    text=text,
                    api_key=self.tts_config["api_key"],
                    voice_name=voice_name,
                    model=self.tts_config["model"],
                    api_url=self.tts_config.get("api_url")
                )
                if pcm_data:
                    yield pcm_data
                else:
                    raise ValueError("Gemini TTS returned empty data")
                    
        except Exception as e:
            logger.error(f"TTS execution failed for '{text[:20]}...': {e}")
            raise
    
    def create_processor(
        self,
        max_concurrent: int = 5,
        max_retry: int = 2,
        task_timeout: float = 30.0,
        first_task_timeout: float = 15.0,
        enable_warmup: bool = True,
    ) -> PredictiveTTSProcessor:
        """
        创建预测性 TTS 处理器
        
        Args:
            max_concurrent: 最大并发数
            max_retry: 最大重试次数
            task_timeout: 任务超时时间
            first_task_timeout: 首句任务超时时间（使用更短超时快速失败）
            enable_warmup: 是否启用 TTS 预热
            
        Returns:
            PredictiveTTSProcessor 实例
        """
        return PredictiveTTSProcessor(
            tts_executor=self.execute_tts,
            max_concurrent=max_concurrent,
            max_retry=max_retry,
            task_timeout=task_timeout,
            first_task_timeout=first_task_timeout,
            enable_warmup=enable_warmup,
        )
    
    def create_streaming_processor(
        self,
        on_audio_chunk: Callable[[bytes], Awaitable[None]],
        max_retry: int = 3,
        task_timeout: float = 60.0,
    ) -> StreamingTTSProcessor:
        """
        创建流式 TTS 处理器（适用于阿里云等平台）
        
        与预测性处理器的区别：
        - 边生成边发送：不等待任务完成
        - 串行处理：保证顺序
        - 实时回调：每个音频块立即发送
        
        Args:
            on_audio_chunk: 音频块回调函数
            max_retry: 最大重试次数
            task_timeout: 任务超时时间
            
        Returns:
            StreamingTTSProcessor 实例
        """
        return StreamingTTSProcessor(
            tts_executor=self.execute_tts,
            on_audio_chunk=on_audio_chunk,
            max_retry=max_retry,
            task_timeout=task_timeout,
        )