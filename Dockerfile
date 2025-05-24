
FROM python:3.9-slim
来自 python: 3.9 -slim

WORKDIR /app
工作目录 /应用程序


COPY requirements.txt .
复制 requirements.txt 。
RUN pip install --no-cache-dir -r requirements.txt
运行 pip install --no-cache-dir -r requirements.txt


COPY ./eztalk_proxy /app/eztalk_proxy
复制 ./eztalk_proxy /app/eztalk_proxy

ENV PORT 7860
环境端口 7860
EXPOSE ${PORT} # 暴露给 Docker，告诉它容器会监听这个端口

CMD uvicorn eztalk_proxy.main:app --host 0.0.0.0 --port ${PORT}
CMD uvicorn eztalk_proxy.main:app --host 0.0.0.0 --port ${PORT}
