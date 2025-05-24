FROM python:3.9-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY ./eztalk_proxy /app/eztalk_proxy

ENV PORT 7860
<<<<<<< HEAD
=======
EXPOSE ${PORT} # 暴露给 Docker，告诉它容器会监听这个端口
>>>>>>> ff55af927926b8ab179f23cec7ead9e49dabe4cf

CMD uvicorn eztalk_proxy.main:app --host 0.0.0.0 --port ${PORT}
CMD uvicorn eztalk_proxy.main:app --host 0.0.0.0 --port ${PORT}
