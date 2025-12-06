import os
import json
import time
import threading
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 图片历史数据存储目录
# 优先使用环境变量 IMAGE_HISTORY_DIR，否则使用默认的绝对路径 /app/data/image_history
# 这确保在 Docker 容器中使用明确的路径，避免相对路径计算问题
_BASE_DIR = os.environ.get("IMAGE_HISTORY_DIR", "/app/data/image_history")

# 如果不在 Docker 环境中（例如本地开发），回退到相对路径
if not os.path.exists("/app") and not _BASE_DIR.startswith("/app"):
    _BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "image_history")

_LOCK = threading.Lock()

# 启动时记录配置的路径
logger.info(f"[IMAGE_STORE] Using image history directory: {_BASE_DIR}")


def _ensure_dir(path: str) -> None:
    """确保目录存在（幂等操作）"""
    os.makedirs(path, exist_ok=True)


def _index_path(conv_id: str) -> str:
    """获取会话的索引文件路径"""
    return os.path.join(_BASE_DIR, conv_id, "index.json")


def save_images(conversation_id: Optional[str], images: List[Dict[str, str]], meta: Optional[Dict[str, Any]] = None) -> None:
    """
    追加保存一批图片记录：
    - conversation_id: 会话ID（必须），为空则忽略
    - images: 形如 [{"url": "..."}]
    - meta: 可选元信息，例如 {"text": "...", "seed": 123, "ts": 1234567890}
    """
    if not conversation_id:
        logger.warning("[IMAGE_STORE] save_images called with empty conversation_id, skipping.")
        return
    if not isinstance(images, list) or not images:
        logger.warning(f"[IMAGE_STORE] save_images called with invalid images for conv={conversation_id}, skipping.")
        return

    ts = int(time.time())
    record = {
        "ts": ts,
        "images": images,
        "meta": meta or {}
    }

    with _LOCK:
        conv_dir = os.path.join(_BASE_DIR, conversation_id)
        _ensure_dir(conv_dir)
        idx_path = _index_path(conversation_id)

        data = []
        if os.path.exists(idx_path):
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
            except Exception as e:
                # 若损坏则重建
                logger.warning(f"[IMAGE_STORE] Failed to read index file for conv={conversation_id}, rebuilding: {e}")
                data = []

        data.append(record)
        try:
            with open(idx_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[IMAGE_STORE] Saved {len(images)} image(s) for conv={conversation_id}, total records: {len(data)}")
        except Exception as e:
            logger.error(f"[IMAGE_STORE] Failed to write index file for conv={conversation_id}: {e}")


def list_images(conversation_id: str) -> List[Dict[str, Any]]:
    """
    返回会话下已保存的所有图像记录（按时间顺序）
    """
    if not conversation_id:
        return []
    idx_path = _index_path(conversation_id)
    if not os.path.exists(idx_path):
        return []
    try:
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"[IMAGE_STORE] Failed to read images for conv={conversation_id}: {e}")
        return []


def delete_conversation(conversation_id: str) -> bool:
    """
    删除会话的持久化记录（不提供递归删除目录，谨慎处理）
    """
    if not conversation_id:
        return False
    idx_path = _index_path(conversation_id)
    try:
        if os.path.exists(idx_path):
            os.remove(idx_path)
            logger.info(f"[IMAGE_STORE] Deleted index file for conv={conversation_id}")
        # 若目录为空，可尝试删除
        conv_dir = os.path.dirname(idx_path)
        try:
            if os.path.isdir(conv_dir) and not os.listdir(conv_dir):
                os.rmdir(conv_dir)
                logger.info(f"[IMAGE_STORE] Removed empty directory for conv={conversation_id}")
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error(f"[IMAGE_STORE] Failed to delete conversation {conversation_id}: {e}")
        return False