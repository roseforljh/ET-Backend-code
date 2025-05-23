
FROM python:3.9-slim


WORKDIR /app

COPY requirements.txt .


# --no-cache-dir 减少镜像大小
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 7860
CMD ["uvicorn", "eztalk_proxy.main:app", "--host", "0.0.0.0", "--port", "80"]
