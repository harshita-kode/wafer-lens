# ---------- WaferLens: API + UI ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf-cache

WORKDIR /app

# 1. Dependencies first: these layers are reused until the requirements change
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

# 2. Code and the files the app needs at runtime
COPY src/ src/
COPY frontend/ frontend/
COPY knowledge_base/ knowledge_base/
COPY samples/ samples/
COPY scripts/build_indexes.py scripts/
COPY data/label_map.json data/
COPY data/processed/wafers_64.npz data/processed/
COPY models/wafer_cnn.pt models/

# 3. Build both vector indexes inside the image (downloads the embedding model once)
RUN python -m scripts.build_indexes

# 4. Run as a non-root user, fully offline except for LLM calls
RUN useradd --create-home appuser && mkdir -p /app/.cache && chown -R appuser /app
USER appuser
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]