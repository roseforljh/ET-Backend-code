# ---- Build Stage ----
FROM python:3.10-slim as builder

WORKDIR /app

# 安装依赖到系统路径
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
FROM python:3.10-slim

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY . .

# 复制并设置 entrypoint 脚本执行权限
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 创建数据目录（确保镜像层中存在，卷挂载时不会被覆盖）
RUN mkdir -p /app/data/image_history

# 暴露端口
EXPOSE 7860

# 使用 entrypoint 脚本启动应用
ENTRYPOINT ["/app/entrypoint.sh"]