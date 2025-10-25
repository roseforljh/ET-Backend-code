import os
import json
import time
import threading
from typing import List, Dict, Any, Optional

# 简单的基于本地文件的持久化：data/image_history/{conversation_id}/index.json
# 仅保存元数据与上游返回的 URL 或 data URI，不主动下载图片，避免阻塞与副作用
_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "image_history")
_LOCK = threading.Lock()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _index_path(conv_id: str) -> str:
    return os.path.join(_BASE_DIR, conv_id, "index.json")


def save_images(conversation_id: Optional[str], images: List[Dict[str, str]], meta: Optional[Dict[str, Any]] = None) -> None:
    """
    追加保存一批图片记录：
    - conversation_id: 会话ID（必须），为空则忽略
    - images: 形如 [{"url": "..."}]
    - meta: 可选元信息，例如 {"text": "...", "seed": 123, "ts": 1234567890}
    """
    if not conversation_id:
        return
    if not isinstance(images, list) or not images:
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
            except Exception:
                # 若损坏则重建
                data = []

        data.append(record)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


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
    except Exception:
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
        # 若目录为空，可尝试删除
        conv_dir = os.path.dirname(idx_path)
        try:
            if os.path.isdir(conv_dir) and not os.listdir(conv_dir):
                os.rmdir(conv_dir)
        except Exception:
            pass
        return True
    except Exception:
        return False