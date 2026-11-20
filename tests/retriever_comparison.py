"""
Retriever strategy comparison (Week 6): which produces better answers?

Every strategy gets the same questions - the answerable, follow-up and
unanswerable cases from tests/data/eval_dataset.json - and the same
answer prompt, so the only thing that differs is what was retrieved.
Follow-ups are rewritten once, up front, so every strategy searches the
same standalone question.

Measured per strategy:
  retrieval  right document among the passages (answerable), nothing
             returned (unanswerable), passage volume, retrieval time
  answers    expected facts present, faithfulness (per-claim judge),
             "not covered" said for unanswerable, answer time

  threshold    similarity with the score/gap gate, no reranker
  similarity   similarity + cross-encoder rerank (the default)
  mmr          maximal marginal relevance + rerank
  multi_query  LLM rephrasings pooled + rerank
  compression  rerank shortlist + per-passage LLM extraction
  parent       similarity + rerank, each match widened to its neighbours

Run:  python -m tests.retriever_comparison      (writes reports/retriever_comparison.*)
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser

from chains.question_rewriter import rewrite_question
from config import settings
from graph.evaluation import facts_present, faithfulness
from graph.rag_workflow import ANSWER, NOT_COVERED, _context
from llm.provider import get_llm, safe_invoke
from rag.metadata import describe
from rag.retriever import retrieve
from rag.vectorstore import load_vectorstore
from tests.rag_eval import DATASET, REPORTS, _history

STRATEGIES = ["threshold", "similarity", "mmr", "multi_query", "compression", "parent"]


def _retrieve(vectorstore, strategy: str, query: str):
    if strategy == "threshold":
        previous = settings.USE_RERANKER
        settings.USE_RERANKER = False
        try:
            return retrieve(vectorstore, query, strategy="similarity")
        finally:
            settings.USE_RERANKER = previous
    return retrieve(vectorstore, query, strategy=strategy)


def main() -> None:
    cases = [c for c in json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
             if c["category"] in ("answerable", "follow_up", "unanswerable")]
    # One rewrite per follow-up, shared by every strategy.
    queries = {c["id"]: rewrite_question(c["question"], _history(c.get("history"))) for c in cases}
    vectorstore = load_vectorstore()
    answer_chain = ANSWER.template() | get_llm() | StrOutputParser()

    rows = []
    for strategy in STRATEGIES:
        print(f"\n== {strategy}", flush=True)
        for case in cases:
            started = time.perf_counter()
            docs = _retrieve(vectorstore, strategy, queries[case["id"]])
            retrieval_s = time.perf_counter() - started
            passages = [{"text": d.page_content.strip(), "source": describe(d)} for d in docs]

            started = time.perf_counter()
            answer = safe_invoke(answer_chain, {"context": _context(passages), "history": [],
                                                "question": queries[case["id"]], "feedback": ""}).strip()
            answer_s = time.perf_counter() - started

            row = {"strategy": strategy, "id": case["id"], "category": case["category"],
                   "passages": len(passages), "context_chars": sum(len(p["text"]) for p in passages),
                   "retrieval_s": round(retrieval_s, 2), "answer_s": round(answer_s, 2)}
            if case["category"] == "unanswerable":
                row["rejected"] = not passages
                row["honest"] = NOT_COVERED.lower() in answer.lower()
            else:
                row["source_hit"] = any(case["source"].lower() in p["source"].lower() for p in passages)
                row["facts_ok"] = facts_present(answer, case["facts"])
                row["faithfulness"] = faithfulness(answer, passages)["score"]
            rows.append(row)
            print(f"   {case['id']:7} passages={row['passages']} {retrieval_s:5.2f}s "
                  + (f"hit={row.get('source_hit')} facts={row.get('facts_ok')} faith={row.get('faithfulness')}"
                     if "source_hit" in row else f"rejected={row['rejected']} honest={row['honest']}"),
                  flush=True)

    summary = {}
    for strategy in STRATEGIES:
        mine = [r for r in rows if r["strategy"] == strategy]
        grounded = [r for r in mine if "source_hit" in r]
        negative = [r for r in mine if "rejected" in r]
        faith = [r["faithfulness"] for r in grounded if r["faithfulness"] is not None]
        summary[strategy] = {
            "source_hit": round(sum(r["source_hit"] for r in grounded) / len(grounded), 3),
            "facts_ok": round(sum(r["facts_ok"] for r in grounded) / len(grounded), 3),
            "faithfulness": round(statistics.mean(faith), 3) if faith else None,
            "correct_rejection": round(sum(r["rejected"] for r in negative) / len(negative), 3),
            "honest_unanswerable": round(sum(r["honest"] for r in negative) / len(negative), 3),
            "context_chars": round(statistics.mean(r["context_chars"] for r in mine)),
            "retrieval_s": round(statistics.median(r["retrieval_s"] for r in mine), 2),
            "answer_s": round(statistics.median(r["answer_s"] for r in mine), 2),
        }

    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model": settings.OLLAMA_MODEL, "summary": summary, "rows": rows}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "retriever_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = ["| Strategy | Right doc | Facts correct | Faithfulness | Rejects unanswerable | Says 'not covered' | Context (chars) | Retrieval p50 | Answer p50 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s, v in summary.items():
        lines.append(f"| {s} | {pct(v['source_hit'])} | {pct(v['facts_ok'])} | {pct(v['faithfulness'])} | "
                     f"{pct(v['correct_rejection'])} | {pct(v['honest_unanswerable'])} | {v['context_chars']} | "
                     f"{v['retrieval_s']}s | {v['answer_s']}s |")
    (REPORTS / "retriever_comparison.md").write_text(
        f"# Retriever comparison - raw results\n\nRun {report['run_at']} · `{report['model']}` · "
        f"{len(cases)} questions per strategy. Generated by `python -m tests.retriever_comparison`; "
        "the analysis is in docs/retrieval_evaluation.md.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
