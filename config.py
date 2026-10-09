import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path = BASE_DIR / ".env") -> None:
    
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


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_path(name: str, default: str) -> str:
    """A filesystem path, always absolute.

    These used to be plain relative strings, which meant they resolved
    against whatever directory the process happened to start in. Launch
    uvicorn from anywhere but the project root and Chroma would create a
    fresh, empty store next to you instead of opening the real one - no
    error, just an assistant that had forgotten every document. It cost
    an hour of debugging a retrieval "bug" that was a working directory.

    An absolute value in the environment is still honoured as given, so
    a deployment can point the store at a mounted volume.
    """
    raw = Path(os.getenv(name, default)).expanduser()
    return str(raw if raw.is_absolute() else (BASE_DIR / raw).resolve())


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


class Settings:
    
    LLM_PROVIDER: str = _get_str("LLM_PROVIDER", "ollama")

    OLLAMA_TEMPERATURE: float = _get_float("OLLAMA_TEMPERATURE", 0.3)

    # 127.0.0.1, not localhost. langchain-ollama defaults to
    # "http://localhost:11434", and on this machine localhost resolves to
    # the IPv6 ::1 while Ollama binds only to IPv4 127.0.0.1 - so every
    # request tried ::1, waited out a ~2s TCP connect timeout, then
    # retried over IPv4 and succeeded. Measured on /api/tags, which does
    # no model work at all: 2056ms via localhost vs 19ms via 127.0.0.1.
    # That ~2s was being paid by every LLM call in the app.
    OLLAMA_BASE_URL: str = _get_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


    # Context window per request - prompt, retrieved chunks, history and the
    # answer all share it. Ollama defaults to 4096 even though qwen2.5
    # supports 32768; at 4096 a long conversation silently loses its oldest
    # turns. Costs VRAM, since the KV cache scales with it.
    OLLAMA_NUM_CTX: int = _get_int("OLLAMA_NUM_CTX", 8192)

    OLLAMA_MODEL: str = _get_str("OLLAMA_MODEL", "qwen2.5:7b")

    # Keep this a non-reasoning model: qwen3:4b measured 4x slower here.
    OLLAMA_MODEL_FAST: str = _get_str("OLLAMA_MODEL_FAST", "qwen2.5:3b")

    ROUTING_WORD_THRESHOLD: int = _get_int("ROUTING_WORD_THRESHOLD", 12)

    REASONING_MODELS: list[str] = _get_list("REASONING_MODELS", "qwen3:4b")

    # --- Embeddings (used by rag/embeddings.py) ---
    EMBEDDING_MODEL: str = _get_str(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    EMBEDDING_DEVICE: str = _get_str("EMBEDDING_DEVICE", "cpu")

    # Load models from the local HuggingFace cache without contacting the
    # Hub. Every SentenceTransformer(...) otherwise checks for updates
    # first, which measured at 9.6s for the cross-encoder against 0.07s
    # offline - the files were local the whole time, the wait was network.
    #
    # Set false to download a model you haven't used before; offline mode
    # fails rather than fetching it.
    HF_OFFLINE: bool = _get_bool("HF_OFFLINE", True)

    # bge models expect an instruction on the query side only. Leave empty
    # for models that don't want one (all-MiniLM, gte) - see rag/embeddings.py.
    EMBEDDING_QUERY_PREFIX: str = _get_str(
        "EMBEDDING_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )

    # --- Vector store / retrieval (used by rag/vectorstore.py, rag/retriever.py) ---
    CHROMA_DB_PATH: str = _get_path("CHROMA_DB_PATH", "chroma_db")
    RETRIEVER_K: int = _get_int("RETRIEVER_K", 3)

    # Relevance gating. See rag/retriever.py for the reasoning and the
    # measurements behind these numbers.
    #
    # The primary test is the *gap*: how far the best match stands above the
    # rest of the candidates. An absolute score cutoff turned out not to
    # work - with bge-small, "for system designing" (irrelevant) scores
    # 0.494 while "goggles" (a direct hit) scores 0.485, so no single cutoff
    # separates them. Their shapes do: gaps of 0.045 and 0.207 respectively.
    #
    # Swept over the 42-query set in tests/retrieval_eval.py. 0.08 is the
    # measured optimum - 33/42 overall, 32/38 excluding small talk.
    #
    # An earlier 0.115 came from sweeping only 13 queries and scored worse
    # on the larger set (30/42): overfitting a tiny sample. Re-sweep with
    # tests/retrieval_eval.py after changing the embedding model, chunking
    # or corpus, and don't tune on a handful of queries.
    #
    # Ceiling is ~79%. The gate handles noise well (17/17 on topically
    # adjacent and unrelated queries) but rejects paraphrased questions
    # (1/7), which genuinely look flat because they share no wording with
    # the source. That needs a cross-encoder reranker, not a better number.
    RETRIEVER_GAP_THRESHOLD: float = _get_float("RETRIEVER_GAP_THRESHOLD", 0.08)

    # How many candidates to score before deciding. Needs to be comfortably
    # more than RETRIEVER_K so the gap is measured against a real spread.
    RETRIEVER_CANDIDATES: int = _get_int("RETRIEVER_CANDIDATES", 10)

    # "similarity" (plain top-k) or "mmr" (Maximum Marginal Relevance - trades
    # some relevance for diversity among the results). Worth it specifically
    # when the corpus has near-duplicate content: two versions of the same
    # lecture PDF meant plain top-6 for "what is a data warehouse" returned
    # the same 3 passages twice. MMR at lambda 0.5 returned 6/6 unique
    # passages instead of 3/6. See rag/retriever.py.
    RETRIEVAL_STRATEGY: str = _get_str("RETRIEVAL_STRATEGY", "similarity")

    # How many candidates MMR considers before picking k diverse ones.
    MMR_FETCH_K: int = _get_int("MMR_FETCH_K", 20)

    # 0 = maximum diversity, 1 = plain relevance ranking (no diversity
    # effect at all). 0.5 is the measured sweet spot on this corpus - 0.9
    # still let through duplicate content, 0.5 and below didn't.
    MMR_LAMBDA: float = _get_float("MMR_LAMBDA", 0.5)

    # How many LLM-generated rephrasings to retrieve for, on top of the
    # original question, when RETRIEVAL_STRATEGY="multi_query".
    MULTI_QUERY_COUNT: int = _get_int("MULTI_QUERY_COUNT", 3)

    # How many of RETRIEVER_CANDIDATES actually go through compression's
    # one-LLM-call-per-chunk extraction, after a cheap cross-encoder pass
    # narrows the field first. Measured on the known diluted html1.md
    # chunk: it ranks 3rd of 10 by cross-encoder score despite failing the
    # absolute threshold, so a shortlist of 4 keeps it in play while
    # cutting LLM calls from 10 to 4 (~5x faster, confirmed to not change
    # the outcome vs running all 10). See rag/compression.py.
    COMPRESSION_CANDIDATES: int = _get_int("COMPRESSION_CANDIDATES", 4)

    # Cross-encoder reranking. Scores each candidate against the question
    # directly instead of comparing embeddings, which is what finally makes
    # paraphrased questions work. See rag/reranker.py.
    USE_RERANKER: bool = _get_bool("USE_RERANKER", True)
    RERANKER_MODEL: str = _get_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Raw logit, not a probability. Swept over tests/retrieval_eval.py:
    # -3.5 scores 37/38 against the gap gate's 32/38.
    RERANKER_SCORE_THRESHOLD: float = _get_float("RERANKER_SCORE_THRESHOLD", -3.5)

    # Secondary net, deliberately low: catches a uniformly terrible set that
    # still happens to have one slightly-less-terrible member. Also the
    # fallback when there are too few candidates to compute a gap from.
    RETRIEVER_SCORE_THRESHOLD: float = _get_float("RETRIEVER_SCORE_THRESHOLD", 0.10)

    # Classify each document at ingest time and store the subject in chunk
    # metadata, so retrieval can later filter by it. Costs one LLM call per
    # upload; turn off if uploads need to be fast.
    CLASSIFY_ON_INGEST: bool = _get_bool("CLASSIFY_ON_INGEST", True)

    # --- Text splitting (used by rag/splitter.py) ---
    # "recursive" (characters), "token", or "recursive_token" (sentence-aware
    # splitting measured in tokens). See rag/splitter.py.
    SPLITTER_TYPE: str = _get_str("SPLITTER_TYPE", "recursive_token")

    # Used when SPLITTER_TYPE is "recursive"
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 100)

    # Used when SPLITTER_TYPE is "token" or "recursive_token".
    # Keep under the embedding model's limit - all-MiniLM-L6-v2 truncates
    # silently above 256 tokens.
    CHUNK_SIZE_TOKENS: int = _get_int("CHUNK_SIZE_TOKENS", 200)
    CHUNK_OVERLAP_TOKENS: int = _get_int("CHUNK_OVERLAP_TOKENS", 40)

    # --- Storage (used by app.py) ---
    UPLOAD_DIR: str = _get_path("UPLOAD_DIR", "uploads")

    # Where thumbs up/down land. One JSON object per line, appended -
    # see feedback.py for why that beats a table here.
    FEEDBACK_PATH: str = _get_path("FEEDBACK_PATH", "feedback.jsonl")

    # --- API (used by app.py) ---
    CORS_ORIGINS: list[str] = _get_list(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )


settings = Settings()


# Applied here rather than at the call site: huggingface_hub reads this
# into a module constant when it is imported, so setting it afterwards has
# no effect. config.py imports no HF code itself, and anything that does
# imports config first, so this runs early enough.
if settings.HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


if __name__ == "__main__":
    for key, value in vars(Settings).items():
        if not key.startswith("_"):
            print(f"{key} = {getattr(settings, key)!r}")
