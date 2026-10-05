"""
The embedding model.

bge models are trained asymmetrically: a question and the passage that
answers it are different kinds of text, so the query gets an instruction
prefix and the documents don't. Skipping the prefix doesn't error - it
just quietly gives up a chunk of the model's advantage, which is the
worst kind of mistake because everything still appears to work.

query_encode_kwargs applies only to embed_query(), so documents are
embedded plain and queries carry the instruction, which is what bge
expects. EMBEDDING_QUERY_PREFIX is empty for models that don't want one
(all-MiniLM, gte), in which case this is a no-op.
"""

from functools import lru_cache

# Imported before langchain_huggingface on purpose - config sets
# HF_HUB_OFFLINE, which huggingface_hub reads at import time.
from config import settings  # isort: skip
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402


@lru_cache(maxsize=1)
def get_embeddings():
    """Cached - building this loads the model, which takes seconds.

    load_vectorstore() calls it too, so without the cache the model was
    built twice on startup: ~8s of pure duplicate work.
    """
    encode_kwargs = {"normalize_embeddings": True}

    query_encode_kwargs = dict(encode_kwargs)
    prefix = settings.EMBEDDING_QUERY_PREFIX
    if prefix:
        # The .env loader strips values, which would eat the trailing space
        # and run the instruction straight into the question
        # ("...passages:what is HTML"). Put it back.
        query_encode_kwargs["prompt"] = prefix if prefix.endswith(" ") else prefix + " "

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs=encode_kwargs,
        query_encode_kwargs=query_encode_kwargs,
    )


if __name__ == "__main__":
    embeddings = get_embeddings()
    print("model :", settings.EMBEDDING_MODEL)
    print("prefix:", repr(settings.EMBEDDING_QUERY_PREFIX))

    vector = embeddings.embed_query("What is HTML?")
    print("dims  :", len(vector))

    # A prefix that is actually being applied changes the query vector
    # while leaving document vectors alone - worth being able to see.
    plain = embeddings.embed_documents(["What is HTML?"])[0]
    same = all(abs(a - b) < 1e-6 for a, b in zip(vector, plain))
    print("query vector differs from document vector:", not same)
