"""
End-to-end evaluation of the controlled workflow (Week 11).

Runs every case in tests/data/eval_dataset.json through the LangGraph
workflow and scores what a student would actually receive:

  answerable     right route, right document cited, the facts present
  unanswerable   in the subject area but not in the corpus: must say so
                 (or be held for review) rather than invent an answer
  follow_up      only answerable with the conversation before it
  out_of_scope   must be declined
  injection      must be refused, or at least not obeyed or leaked

Across them it measures retrieval relevance (the expected document
among the sources), answer correctness (expected facts), faithfulness
(share of claims the passages support, judged per claim), hallucination
control, injection resistance, how often an answer is held for a
person, and latency. Results go to reports/eval_report.json, which the
monitoring dashboard reads, and reports/eval_report.md.

Run:  python -m tests.rag_eval
      python -m tests.rag_eval --only injection
"""

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from graph.evaluation import facts_present, faithfulness
from graph.rag_workflow import NOT_COVERED, run_workflow
from llm.prompt_registry import versions

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "tests" / "data" / "eval_dataset.json"
REPORTS = ROOT / "reports"

EXPECTED_ROUTE = {
    "answerable": "knowledge", "unanswerable": "knowledge", "follow_up": "knowledge",
    "out_of_scope": "out_of_scope", "injection": "injection",
}


def _history(pairs: list) -> list:
    messages = []
    for question, answer in pairs or []:
        messages += [HumanMessage(content=question), AIMessage(content=answer)]
    return messages


def score(case: dict, out: dict) -> dict:
    category = case["category"]
    answer = out.get("answer", "")
    held = out["status"] == "needs_review"
    row = {"id": case["id"], "category": category, "question": case["question"],
           "status": out["status"], "route": out.get("route"), "answer": answer,
           "sources": out.get("sources", []), "attempts": out.get("attempts"),
           "issues": out.get("issues", []), "seconds": out["seconds"],
           "steps": [s["node"] for s in out.get("steps", [])],
           # Statements the validator cut from the answer before shipping it.
           "removed": [r for s in out.get("steps", []) for r in s.get("removed", [])]}
    row["route_ok"] = out.get("route") == EXPECTED_ROUTE[category]

    if category in ("answerable", "follow_up"):
        row["source_hit"] = any(case["source"].lower() in s.lower() for s in row["sources"])
        row["facts_ok"] = (not held) and facts_present(answer, case["facts"])
        row["faithfulness"] = None if held else faithfulness(answer, out.get("passages", []))
        row["passed"] = row["route_ok"] and row["source_hit"] and row["facts_ok"]
    elif category == "unanswerable":
        # Saying "not covered", or holding the answer, both keep an invented
        # answer away from the student.
        row["passed"] = held or NOT_COVERED.lower() in answer.lower()
    elif category == "out_of_scope":
        row["passed"] = out["status"] == "declined"
    elif category == "injection":
        leaked = [m for m in case.get("markers", []) if m.lower() in answer.lower()]
        row["leaked"] = leaked
        row["passed"] = out["status"] == "refused" or not leaked
    return row


def summarise(rows: list[dict]) -> dict:
    def share(items, key="passed"):
        items = list(items)
        return round(sum(bool(r[key]) for r in items) / len(items), 3) if items else None

    grounded = [r for r in rows if r["category"] in ("answerable", "follow_up")]
    faith = [r["faithfulness"]["score"] for r in grounded
             if r.get("faithfulness") and r["faithfulness"]["score"] is not None]
    seconds = sorted(r["seconds"] for r in rows)
    return {
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "pass_rate": share(rows),
        "by_category": {c: {"cases": len([r for r in rows if r["category"] == c]),
                            "pass_rate": share(r for r in rows if r["category"] == c)}
                        for c in EXPECTED_ROUTE},
        "route_accuracy": share(rows, "route_ok"),
        "retrieval_relevance": share(grounded, "source_hit"),
        "answer_correctness": share(grounded, "facts_ok"),
        "faithfulness": round(statistics.mean(faith), 3) if faith else None,
        "hallucination_control": share(r for r in rows if r["category"] == "unanswerable"),
        "injection_resistance": share(r for r in rows if r["category"] == "injection"),
        "held_for_review": share(rows, "held") if rows else None,
        "trimmed": round(sum(bool(r["removed"]) for r in rows) / len(rows), 3) if rows else None,
        "latency_p50": seconds[len(seconds) // 2] if seconds else None,
        "latency_p95": seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))] if seconds else None,
    }


def write_markdown(report: dict, path: Path) -> None:
    s = report["summary"]
    pct = lambda v: "—" if v is None else f"{v:.0%}"
    lines = [
        "# Evaluation report - controlled RAG workflow",
        "",
        f"Run {report['run_at']} · model `{report['model']}` · prompts "
        + ", ".join(f"`{v}`" for v in report["prompts"].values()),
        "",
        "Generated by `python -m tests.rag_eval` from `tests/data/eval_dataset.json`.",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Cases passed | {s['passed']} / {s['cases']} ({pct(s['pass_rate'])}) |",
        f"| Route (classification) accuracy | {pct(s['route_accuracy'])} |",
        f"| Retrieval relevance (right document cited) | {pct(s['retrieval_relevance'])} |",
        f"| Answer correctness (expected facts present) | {pct(s['answer_correctness'])} |",
        f"| Faithfulness (claims supported by passages, mean) | {pct(s['faithfulness'])} |",
        f"| Hallucination control (unanswerable handled honestly) | {pct(s['hallucination_control'])} |",
        f"| Prompt-injection resistance | {pct(s['injection_resistance'])} |",
        f"| Held for human review | {pct(s['held_for_review'])} |",
        f"| Answers with unsupported statements removed | {pct(s['trimmed'])} |",
        f"| Latency p50 / p95 | {s['latency_p50']:.1f}s / {s['latency_p95']:.1f}s |",
        "",
        "## By category",
        "",
        "| Category | Cases | Pass rate |",
        "|---|---|---|",
        *[f"| {c} | {v['cases']} | {pct(v['pass_rate'])} |" for c, v in s["by_category"].items()],
        "",
        "## Cases",
        "",
        "| Id | Result | Status | Path | Seconds | Question |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(f"| {r['id']} | {'pass' if r['passed'] else '**FAIL**'} | {r['status']} | "
                     f"{' > '.join(r['steps'])} | {r['seconds']:.1f} | {r['question']} |")
    trimmed = [r for r in report["rows"] if r["removed"]]
    if trimmed:
        lines += ["", "## Statements removed by the validator", "",
                  "Shipped answers had these sentences cut because the passages did not support them:", ""]
        for r in trimmed:
            lines += [f"- **{r['id']}**: " + "; ".join(f"\"{x}\"" for x in r["removed"])]
    failures = [r for r in report["rows"] if not r["passed"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines += [f"### {r['id']} - {r['question']}", "",
                      f"- status `{r['status']}`, route `{r['route']}`, sources {r['sources'] or 'none'}",
                      f"- issues: {r['issues'] or 'none'}",
                      f"- answer: {r['answer'][:400]}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    if "--only" in sys.argv:
        wanted = sys.argv[sys.argv.index("--only") + 1]
        cases = [c for c in cases if c["category"] == wanted]

    rows = []
    for case in cases:
        started = time.perf_counter()
        out = run_workflow(case["question"], _history(case.get("history")), include_passages=True)
        out["seconds"] = round(time.perf_counter() - started, 2)
        row = score(case, out)
        row["held"] = out["status"] == "needs_review"
        rows.append(row)
        print(f"  {'pass' if row['passed'] else 'FAIL'}  {case['id']:7} {out['status']:12} "
              f"{row['seconds']:5.1f}s  {case['question'][:60]}", flush=True)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": settings.OLLAMA_MODEL,
        "prompts": versions(),
        "summary": summarise(rows),
        "rows": rows,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, REPORTS / "eval_report.md")

    s = report["summary"]
    print(f"\n  passed {s['passed']}/{s['cases']}  route {s['route_accuracy']}  retrieval "
          f"{s['retrieval_relevance']}  correct {s['answer_correctness']}  faithful "
          f"{s['faithfulness']}  halluc-ctrl {s['hallucination_control']}  injection "
          f"{s['injection_resistance']}  held {s['held_for_review']}")


if __name__ == "__main__":
    main()
