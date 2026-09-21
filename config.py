"""
Central environment configuration for the AI Knowledge Assistant.

Every value that used to be hardcoded across rag/, llm/, and app.py is
defined here, with a sensible default and an override via .env / real
environment variables. Nothing here requires extra dependencies -
just the standard library - so it works regardless of what's installed
in whichever Python environment ends up running the app.

Usage:
    from config import settings
    settings.OLLAMA_MODEL
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    """Minimal .env loader (stdlib only).

    Reads KEY=VALUE lines into os.environ. Real environment variables
    (e.g. set by the shell or a deployment platform) always win - a
    value already present in os.environ is never overwritten.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    # --- LLM provider (used by llm/provider.py) ---
    # Only "ollama" is supported - both models below are free, open-weight,
    # and run locally, so routing between them costs nothing and needs no API key.
    LLM_PROVIDER: str = _get_str("LLM_PROVIDER", "ollama")

    OLLAMA_TEMPERATURE: float = _get_float("OLLAMA_TEMPERATURE", 0.3)

    # General-purpose model: used for longer / more complex questions.
    OLLAMA_MODEL: str = _get_str("OLLAMA_MODEL", "qwen2.5:7b")

    # Fast model: used for short / simple questions, to avoid paying the
    # larger model's load and inference cost when it isn't needed.
    OLLAMA_MODEL_FAST: str = _get_str("OLLAMA_MODEL_FAST", "qwen3:4b")

    # Routing heuristic: questions with <= this many words are considered
    # "simple" and go to OLLAMA_MODEL_FAST; longer ones go to OLLAMA_MODEL.
    ROUTING_WORD_THRESHOLD: int = _get_int("ROUTING_WORD_THRESHOLD", 12)

    # --- Embeddings (used by rag/embeddings.py) ---
    EMBEDDING_MODEL: str = _get_str(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DEVICE: str = _get_str("EMBEDDING_DEVICE", "cpu")

    # --- Vector store / retrieval (used by rag/vectorstore.py, rag/retriever.py) ---
    CHROMA_DB_PATH: str = _get_str("CHROMA_DB_PATH", "./chroma_db")
    RETRIEVER_K: int = _get_int("RETRIEVER_K", 3)

    # --- Text splitting (used by rag/splitter.py) ---
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 100)

    # --- Storage (used by app.py) ---
    UPLOAD_DIR: str = _get_str("UPLOAD_DIR", "uploads")

    # --- API (used by app.py) ---
    CORS_ORIGINS: list[str] = _get_list(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )


settings = Settings()


if __name__ == "__main__":
    for key, value in vars(Settings).items():
        if not key.startswith("_"):
            print(f"{key} = {getattr(settings, key)!r}")
