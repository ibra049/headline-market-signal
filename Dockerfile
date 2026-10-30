FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts/champion

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt \
    && useradd --create-home --uid 1000 app

COPY --chown=app:app serve/ serve/
COPY --chown=app:app src/hms/onnx_predict.py src/hms/onnx_predict.py
COPY --chown=app:app artifacts/champion/ artifacts/champion/

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
