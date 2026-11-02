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
# The champion model lives in the public Hugging Face model repo (promoted by the CI eval gate).
ARG HF_MODEL_REPO=Ibra-049/headline-sentiment
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${HF_MODEL_REPO}', local_dir='artifacts/champion')" \
    && chown -R app:app artifacts

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{__import__("os").environ.get("PORT", "7860")}/health', timeout=4)"

# Render sets $PORT; 7860 locally.
CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
