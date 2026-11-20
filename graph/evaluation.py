"""Scoring a workflow answer: faithfulness, correctness, leaks.

Faithfulness is judged claim by claim, by a prompt of its own rather than
the workflow's validator. The validator decides whether to ship an answer;
this measures the answer that shipped. Using one for both would mark the
workflow's homework with its own answer key. It is still a model of the
same family judging, which is the known limit of an LLM judge - the
per-claim breakdown is kept in the report so a person can audit it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from graph.rag_workflow import GENERAL_LABEL, _context, _is_claim
from llm.prompt_registry import register
from llm.provider import get_llm, safe_invoke

FAITHFULNESS = register("eval.faithfulness", "v1", [
    ("system",
     "Split the answer into its individual factual claims. For each claim, "
     "decide whether the passages support it: stated by a passage, or "
     "following directly from one. Do not use your own knowledge - a true "
     "claim the passages do not make is unsupported."),
    ("human", "Passages:\n{context}\n\nAnswer:\n{answer}"),
])


class _Claim(BaseModel):
    claim: str
    supported: bool


class _Claims(BaseModel):
    claims: list[_Claim] = Field(description="Every factual claim in the answer, judged")


def faithfulness(answer: str, passages: list[dict]) -> dict:
    """Share of the answer's claims the passages support, with the claims.

    Only the part that claims to come from the materials is judged; a
    labelled general-knowledge addendum makes no such claim.
    """
    claimed = (answer or "").split(GENERAL_LABEL)[0].strip()
    if not passages or not claimed:
        return {"score": None, "claims": []}
    raw = safe_invoke(FAITHFULNESS.template() | get_llm().with_structured_output(_Claims),
                      {"context": _context(passages), "answer": claimed})
    claims = [c for c in (raw.claims if raw else []) if _is_claim(c.claim)]
    if not claims:
        return {"score": None, "claims": []}
    return {
        "score": round(sum(c.supported for c in claims) / len(claims), 3),
        "claims": [{"claim": c.claim, "supported": c.supported} for c in claims],
    }


def facts_present(answer: str, facts: list) -> bool:
    """Every fact appears; a list entry passes if any alternative does."""
    text = (answer or "").lower()
    return all(
        any(alt.lower() in text for alt in (fact if isinstance(fact, list) else [fact]))
        for fact in facts
    )
