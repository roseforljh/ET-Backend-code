FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY ./eztalk_proxy /app/eztalk_proxy


ENV PORT 7860 
EXPOSE ${PORT} 


CMD uvicorn eztalk_proxy.main:app --host 0.0.0.0 --port ${PORT}