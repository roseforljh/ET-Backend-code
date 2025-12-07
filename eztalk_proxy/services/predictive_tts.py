"""
预测性 TTS 处理器

实现 LLM 输出与 TTS 合成的并行处理，减少语音响应延迟。

核心思想：
- LLM 输出片段后立即提交 TTS 任务
- TTS 任务在后台并行执行
- 按顺序输出音频，确保播放连贯性
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Awaitable
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
    - 支持 Gemini、Minimax、SiliconFlow、OpenAI 四种平台
    """
    
    def __init__(
        self,
        tts_executor: Callable[[str], Awaitable[AsyncGenerator[bytes, None]]],
        max_concurrent: int = 5,
        max_retry: int = 2,
        task_timeout: float = 30.0
    ):
        """
        初始化预测性 TTS 处理器
        
        Args:
            tts_executor: TTS 执行函数，接收文本，返回音频流
            max_concurrent: 最大并发 TTS 请求数
            max_retry: 最大重试次数
            task_timeout: 单任务超时时间（秒）
        """
        self.tts_executor = tts_executor
        self.max_concurrent = max_concurrent
        self.max_retry = max_retry
        self.task_timeout = task_timeout
        
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
        
        logger.info(f"PredictiveTTSProcessor initialized: max_concurrent={max_concurrent}, max_retry={max_retry}")
    
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
            
        task = TTSTask(sequence_id=sequence_id, text=text)
        self.tasks[sequence_id] = task
        self.total_tasks = max(self.total_tasks, sequence_id + 1)
        
        # 记录详细的任务信息
        logger.info(f"[TTS Task {sequence_id}] 提交任务: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # 启动处理协程
        async_task = asyncio.create_task(self._process_task_with_retry(task))
        self._active_tasks.append(async_task)
    
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
    
    async def _execute_tts(self, task: TTSTask) -> None:
        """
        执行 TTS 合成
        
        Args:
            task: TTS 任务
        """
        task.audio_chunks = []
        
        # 使用超时包装
        async def collect_audio():
            async for chunk in self.tts_executor(task.text):
                if chunk:
                    task.audio_chunks.append(chunk)
        
        await asyncio.wait_for(collect_audio(), timeout=self.task_timeout)
    
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
        task_timeout: float = 30.0
    ) -> PredictiveTTSProcessor:
        """
        创建预测性 TTS 处理器
        
        Args:
            max_concurrent: 最大并发数
            max_retry: 最大重试次数
            task_timeout: 任务超时时间
            
        Returns:
            PredictiveTTSProcessor 实例
        """
        return PredictiveTTSProcessor(
            tts_executor=self.execute_tts,
            max_concurrent=max_concurrent,
            max_retry=max_retry,
            task_timeout=task_timeout
        )