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

# 暴露端口
EXPOSE 7860

# 运行应用 (使用 python -m uvicorn 更健壮)
CMD ["python", "-m", "uvicorn", "eztalk_proxy.main:app", "--host", "0.0.0.0", "--port", "7860"]