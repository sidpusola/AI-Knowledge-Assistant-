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

import statistics

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from config import settings
from llm.provider import is_small_talk


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


def _mmr_candidates(
    vectorstore, query: str, where: dict | None
) -> list[Document]:
    """Diversity-aware candidates - no relevance scores come back from this,
    which is why it's handled as its own branch in retrieve() rather than
    folded into the scored similarity path below.
    """
    return vectorstore.max_marginal_relevance_search(
        query,
        k=settings.RETRIEVER_CANDIDATES,
        fetch_k=settings.MMR_FETCH_K,
        lambda_mult=settings.MMR_LAMBDA,
        filter=where,
    )


def _multi_query_candidates(
    vectorstore, query: str, where: dict | None
) -> list[Document]:
    """Candidates pooled from the original question plus a few LLM-generated
    rephrasings, deduplicated by content. No relevance scores either - each
    phrasing has its own, not comparable across phrasings without a
    reranker to score them all against the one real question.
    """
    from rag.multi_query import generate_query_variants

    pooled: dict[str, Document] = {}
    for variant in generate_query_variants(query):
        for doc in vectorstore.similarity_search(
            variant, k=settings.RETRIEVER_CANDIDATES, filter=where
        ):
            pooled.setdefault(doc.page_content, doc)

    return list(pooled.values())


def _rerank_and_gate(query: str, candidates: list[Document], k: int) -> list[Document]:
    """Shared by every strategy that produces unscored candidates (MMR,
    multi-query): the cross-encoder is what actually judges them against
    the real question, regardless of how they were gathered.
    """
    from rag.reranker import rerank

    ranked = rerank(query, candidates)
    kept = [doc for doc, score in ranked if score >= settings.RERANKER_SCORE_THRESHOLD]
    return kept[:k]


def retrieve(
    vectorstore,
    query: str,
    *,
    category: str | None = None,
    file_type: str | None = None,
    k: int | None = None,
    strategy: str | None = None,
) -> list[Document]:
    """Retrieve chunks, or nothing if none are convincingly relevant.

    An empty list is a real outcome: the tutor prompt answers from general
    knowledge and says so, which beats grounding in an unrelated document.
    """
    if is_small_talk(query):
        return []

    k = k or settings.RETRIEVER_K
    strategy = strategy or settings.RETRIEVAL_STRATEGY
    where = _build_filter(category, file_type)

    if strategy in ("mmr", "multi_query"):
        candidates = (
            _mmr_candidates(vectorstore, query, where)
            if strategy == "mmr"
            else _multi_query_candidates(vectorstore, query, where)
        )
        if not candidates:
            return []

        if settings.USE_RERANKER:
            return _rerank_and_gate(query, candidates, k)

        # Neither strategy produces relevance scores of its own, so without
        # a reranker there is no "is this actually relevant" signal here -
        # just whatever came back. Known gap: an irrelevant query still
        # returns k chunks in this combination.
        return candidates[:k]

    if strategy == "compression":
        # The extraction step (does this chunk have anything relevant,
        # yes/no) is itself the relevance judgement, but running it on all
        # RETRIEVER_CANDIDATES means one LLM call per candidate - measured
        # at ~0.3-0.85s each, ~4-8s per query. A cheap cross-encoder pass
        # first narrows the field: the known diluted html1.md chunk ranks
        # 3rd of 10 by cross-encoder score despite failing the absolute
        # threshold, so a shortlist keeps it in play while cutting LLM
        # calls from 10 to COMPRESSION_CANDIDATES - measured ~5x faster
        # with no change in outcome. See rag/compression.py.
        from rag.compression import compress_candidates
        from rag.reranker import rerank

        candidates = vectorstore.similarity_search(
            query, k=settings.RETRIEVER_CANDIDATES, filter=where
        )
        if not candidates:
            return []

        ranked = rerank(query, candidates)
        shortlist = [doc for doc, _ in ranked[: settings.COMPRESSION_CANDIDATES]]

        return compress_candidates(query, shortlist)[:k]

    scored = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=settings.RETRIEVER_CANDIDATES,
        filter=where,
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
    strategy: str | None = None,
):
    """Runnable wrapper so chains can use retrieve() with `|`.

    Not as_retriever(): that only offers an absolute score threshold,
    which is the thing that doesn't work on this data.
    """
    def _run(query: str) -> list[Document]:
        return retrieve(
            vectorstore, query, category=category, file_type=file_type, k=k, strategy=strategy
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

    print("\nsimilarity vs MMR - this corpus has two versions of the same")
    print("data warehousing lecture, which plain top-k returns duplicates of:\n")

    query = "what is a data warehouse"
    for strategy in ("similarity", "mmr"):
        docs = retrieve(vectorstore, query, strategy=strategy, k=6)
        unique = len({d.page_content for d in docs})
        print(f"  {strategy:11} -> {unique}/{len(docs)} chunks have distinct content")
        for document in docs:
            print(f"        {describe(document)}")
