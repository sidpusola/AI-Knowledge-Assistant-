"""
Multi-query retrieval - ask the LLM to rephrase the question a few
different ways, retrieve for each phrasing, and pool the unique results
before the usual reranking/gating runs.

The idea: a single embedding of one phrasing can miss a chunk that uses
different words for the same thing. Several phrasings, each embedded and
searched separately, cover more of the ways the answer might be worded.

This is the one retrieval strategy in this module that costs an LLM call
before retrieval even starts - roughly a second, against ~85ms for the
cross-encoder path. Worth knowing whether it earns that cost: the
reranker already fixed every paraphrase in tests/retrieval_eval.py except
one ("why should I use meaningful tags instead of div everywhere" still
scores -8.889, well below threshold). If multi-query doesn't recover that
one either, it isn't solving a problem the reranker left unsolved here -
see the __main__ block, which checks this directly rather than assuming.
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from llm.provider import get_llm, safe_invoke

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
A student asked a question. Write {n} different ways of asking the same
question, using different words and sentence structure each time.

Keep each one a complete, standalone question - someone reading just one
of them, without seeing the original, should understand what's being
asked. Don't answer the question, only rephrase it.
""",
    ),
    ("human", "{question}"),
])


class QueryVariants(BaseModel):
    queries: list[str] = Field(description="Alternate phrasings of the question")


def generate_query_variants(question: str, n: int | None = None) -> list[str]:
    """The original question plus n LLM-generated rephrasings, deduplicated.

    The original is always included and always first - a rephrasing is a
    supplement to the literal question, not a replacement for it.
    """
    n = n or settings.MULTI_QUERY_COUNT

    llm = get_llm().with_structured_output(QueryVariants)
    chain = _prompt | llm
    result = safe_invoke(chain, {"question": question, "n": n})

    seen: set[str] = set()
    variants: list[str] = []
    for q in [question, *result.queries]:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            variants.append(q)

    return variants


if __name__ == "__main__":
    import time

    from rag.reranker import rerank
    from rag.service import service

    # The one query the cross-encoder gate still rejects (score -8.889,
    # threshold -3.5) despite the corpus covering it directly.
    question = "why should I use meaningful tags instead of div everywhere"

    print(f"question: {question!r}\n")

    t0 = time.perf_counter()
    variants = generate_query_variants(question)
    print(f"generated {len(variants)} variants in {time.perf_counter()-t0:.2f}s:")
    for v in variants:
        print(f"  - {v}")
    print()

    pooled: dict[str, object] = {}
    for variant in variants:
        for doc in service.vectorstore.similarity_search(variant, k=10):
            pooled.setdefault(doc.page_content, doc)

    candidates = list(pooled.values())
    print(f"{len(candidates)} unique candidates pooled from {len(variants)} queries\n")

    ranked = rerank(question, candidates)
    print("reranked against the ORIGINAL question (top 3):")
    for doc, score in ranked[:3]:
        passed = "PASS" if score >= settings.RERANKER_SCORE_THRESHOLD else "below threshold"
        print(f"  {score:8.3f}  [{passed}]  {doc.page_content[:60]!r}")
