"""
Cross-encoder reranking.

The retriever compares two independently-computed embeddings, so it can
only see whether a question and a chunk land near each other in vector
space. That works when they share vocabulary and fails when they don't -
"how is data organised for business reporting" and a chunk explaining
data warehousing mean the same thing but share almost no words, so the
bi-encoder ranks it no better than noise.

A cross-encoder reads the question and the chunk *together* in one pass
and scores the pair directly. It can't be precomputed or indexed, which
is why it isn't used for search itself - but over 10 candidates it costs
about 12ms, and it is dramatically better at the judgement the gate needs.

Measured over the 38 non-chatty queries in tests/retrieval_eval.py:

                            correct   false neg   false pos
    bi-encoder gap           32/38        4           2
    cross-encoder (-3.5)     37/38        1           0

Scores are raw logits, roughly -11 to +11, not probabilities. The useful
range here is narrow and entirely negative: the strongest irrelevant
query scored -3.803 and the weakest genuine one -3.375, so -3.5 sits
between them. That looks precarious written down, but everything else is
far from the boundary - the next irrelevant query is at -4.684 and most
genuine ones are positive.
"""

from functools import lru_cache

from langchain_core.documents import Document

from config import settings


@lru_cache(maxsize=1)
def get_cross_encoder():
    """Loaded once per process - construction reads the model from disk."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.RERANKER_MODEL, device=settings.EMBEDDING_DEVICE)


def rerank(
    query: str, documents: list[Document], *, top_k: int | None = None
) -> list[tuple[Document, float]]:
    """Score each document against the query and sort by that score.

    Returns (document, score) pairs, highest first.
    """
    if not documents:
        return []

    scores = get_cross_encoder().predict([(query, d.page_content) for d in documents])
    ranked = sorted(zip(documents, (float(s) for s in scores)), key=lambda p: -p[1])

    return ranked[:top_k] if top_k else ranked
