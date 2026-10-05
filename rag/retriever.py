"""
Retrieval - similarity search, metadata filtering, and a relevance gate.

An absolute score cutoff does not work here. Measured with bge-small:

    "for system designing"  (irrelevant)   top 0.494   gap 0.045
    "goggles"               (direct hit)   top 0.485   gap 0.207

The irrelevant query scores higher, so no cutoff separates them. The
shape of the distribution does: a query the corpus can answer has one
chunk standing clear of the rest, a query it can't returns ten roughly
equal mediocre matches.

Measured gaps - genuine 0.127-0.253, noise 0.027-0.055.

Limitation: this assumes one chunk should win. A question genuinely
answered by several chunks equally looks flat and gets rejected.
"""

import re
import statistics

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from config import settings

# Greetings and pleasantries shouldn't reach the vector store at all. They
# aren't just irrelevant - they score deceptively well, because a short
# social phrase spikes against some arbitrary chunk. In the eval set
# "thanks for your help" had a gap of 0.291, higher than almost every
# genuine question, so the relevance gate waves it straight through.
_SMALL_TALK = {
    "hello", "hi", "hey", "yo", "hiya", "greetings",
    "how are you", "how are you doing", "hows it going", "how is it going",
    "whats up", "what's up",
    "thanks", "thank you", "thanks a lot", "thank you so much", "cheers",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "see you later",
    "ok", "okay", "cool", "nice", "great", "sure",
    "can you help me", "can you help", "help me", "i need help",
}

# A real question in the text means it isn't small talk, however politely
# it opens - "thanks, what is HTML?" must still retrieve.
_QUESTION_MARKERS = {
    "what", "how", "why", "when", "where", "which", "who", "whose",
    "explain", "define", "describe", "list", "compare", "difference",
    "tell", "show", "give", "is", "are", "does", "do", "can", "should",
}

_PLEASANTRY_OPENERS = (
    "thanks", "thank you", "hello", "hi ", "hey ", "good morning",
    "good afternoon", "good evening", "bye", "goodbye",
)


def is_small_talk(query: str) -> bool:
    """True for greetings and pleasantries carrying no question."""
    cleaned = re.sub(r"[^\w\s]", "", query.lower()).strip()

    # Exact matches first, so "how are you" isn't caught by the "how"
    # question marker below.
    if cleaned in _SMALL_TALK:
        return True

    words = cleaned.split()

    if any(word in _QUESTION_MARKERS for word in words):
        return False

    return len(words) <= 5 and cleaned.startswith(_PLEASANTRY_OPENERS)


def _build_filter(category: str | None, file_type: str | None) -> dict | None:
    """Chroma's where-filter. Two or more conditions need $and."""
    conditions = []

    if category:
        conditions.append({"category": category})
    if file_type:
        conditions.append({"file_type": file_type})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search_with_scores(
    vectorstore,
    query: str,
    *,
    category: str | None = None,
    file_type: str | None = None,
    k: int | None = None,
):
    """Ungated search returning (document, score) pairs, for inspection."""
    return vectorstore.similarity_search_with_relevance_scores(
        query,
        k=k or settings.RETRIEVER_K,
        filter=_build_filter(category, file_type),
    )


def relevance_gap(scores: list[float]) -> float:
    """How far the best score stands above the mean of the rest.

    0.0 when there's nothing to compare against - read as "can't judge",
    not "irrelevant".
    """
    if len(scores) < 2:
        return 0.0
    return scores[0] - statistics.mean(scores[1:])


def retrieve(
    vectorstore,
    query: str,
    *,
    category: str | None = None,
    file_type: str | None = None,
    k: int | None = None,
) -> list[Document]:
    """Retrieve chunks, or nothing if none are convincingly relevant.

    An empty list is a real outcome: the tutor prompt answers from general
    knowledge and says so, which beats grounding in an unrelated document.
    """
    if is_small_talk(query):
        return []

    k = k or settings.RETRIEVER_K

    scored = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=settings.RETRIEVER_CANDIDATES,
        filter=_build_filter(category, file_type),
    )

    if not scored:
        return []

    # The bi-encoder's job ends here - it narrowed the corpus to a shortlist.
    # Judging that shortlist is the cross-encoder's, and it's much better at
    # it, particularly for questions phrased differently from the source.
    if settings.USE_RERANKER:
        from rag.reranker import rerank

        ranked = rerank(query, [doc for doc, _ in scored])
        kept = [doc for doc, score in ranked if score >= settings.RERANKER_SCORE_THRESHOLD]
        return kept[:k]

    scores = [score for _, score in scored]

    # Too few candidates to judge a shape - fall back to the absolute floor.
    # Happens on a narrow filter or a nearly empty collection.
    if len(scored) < 3:
        return [d for d, s in scored[:k] if s >= settings.RETRIEVER_SCORE_THRESHOLD]

    # A uniformly terrible set can still have a standout member.
    if scores[0] < settings.RETRIEVER_SCORE_THRESHOLD:
        return []

    if relevance_gap(scores) < settings.RETRIEVER_GAP_THRESHOLD:
        return []

    return [d for d, _ in scored[:k]]


def get_retriever(
    vectorstore,
    *,
    category: str | None = None,
    file_type: str | None = None,
    k: int | None = None,
):
    """Runnable wrapper so chains can use retrieve() with `|`.

    Not as_retriever(): that only offers an absolute score threshold,
    which is the thing that doesn't work on this data.
    """
    def _run(query: str) -> list[Document]:
        return retrieve(
            vectorstore, query, category=category, file_type=file_type, k=k
        )

    return RunnableLambda(_run)


if __name__ == "__main__":
    from rag.metadata import describe
    from rag.vectorstore import load_vectorstore

    vectorstore = load_vectorstore()
    print(
        f"gap>={settings.RETRIEVER_GAP_THRESHOLD}  "
        f"floor>={settings.RETRIEVER_SCORE_THRESHOLD}  "
        f"candidates={settings.RETRIEVER_CANDIDATES}\n"
    )

    for query in [
        "what is semantic HTML",
        "what is a data warehouse",
        "ETL process",
        "tell me about cse job roles",
        "for system designing",
        "what is the capital of France",
    ]:
        scored = search_with_scores(vectorstore, query, k=settings.RETRIEVER_CANDIDATES)
        scores = [s for _, s in scored]
        kept = retrieve(vectorstore, query)

        verdict = f"{len(kept)} chunk(s)" if kept else "REJECTED"
        print(
            f"  {query[:32]:34} top={scores[0]:6.3f}  "
            f"gap={relevance_gap(scores):6.3f}  -> {verdict}"
        )
        for document in kept:
            print(f"        {describe(document)}")
