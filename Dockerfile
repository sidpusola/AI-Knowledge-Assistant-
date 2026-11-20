# Backend: FastAPI + LangChain + LangGraph + ChromaDB.
# The model server (Ollama) and the web front end are separate services;
# see docker-compose.yml.

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU build of torch first: the default wheel bundles CUDA and is several
# gigabytes larger, for a container that does embeddings on the CPU.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model and the reranker into the image so the
# container starts without a network check and runs fully offline.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')"

COPY . .

# Not root. Everything the app writes goes under /data, a volume.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data \
 && chown -R app /data /opt/hf
USER app

ENV HF_OFFLINE=true \
    DATABASE_URL=sqlite:////data/app.db \
    CHROMA_DB_PATH=/data/chroma_db \
    UPLOAD_DIR=/data/uploads \
    FEEDBACK_PATH=/data/feedback.jsonl \
    WORKFLOW_DB_PATH=/data/workflow_checkpoints.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')"

# --proxy-headers: behind nginx, the original scheme and client address
# come from X-Forwarded-* headers.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
