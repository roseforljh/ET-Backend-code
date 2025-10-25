FROM python:3.9-slim-buster as builder

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.9-slim-buster as final

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
ENV PATH=$APP_HOME/.local/bin:$PATH
ENV TEMP_UPLOAD_DIR=${APP_HOME}/temp_document_uploads

WORKDIR ${APP_HOME}

RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

RUN mkdir -p ${TEMP_UPLOAD_DIR} && \
    chown -R appuser:appgroup ${TEMP_UPLOAD_DIR} && \
    chmod -R 750 ${TEMP_UPLOAD_DIR}

COPY --from=builder /root/.local /home/appuser/.local

COPY eztalk_proxy ./eztalk_proxy
COPY run.py .
COPY deployment/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh && \
    chown appuser:appgroup /entrypoint.sh

RUN chown -R appuser:appgroup /home/appuser/.local

EXPOSE 7860

USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "run.py"]