#!/bin/bash
set -e

# 确保数据目录存在（幂等操作，目录已存在时不会报错也不会删除内容）
mkdir -p /app/data/image_history
mkdir -p /tmp/temp_document_uploads

echo "[entrypoint] Data directories ensured."

# 启动应用
exec python -m uvicorn eztalk_proxy.main:app --host 0.0.0.0 --port 7860