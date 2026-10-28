"""
Agent evaluation set.

Checks one thing: given a question, does the agent reach for the right
tool. Not answer quality - the RAG eval already covers whether the
right passages come back, and answer wording is not mechanically
checkable. Tool choice is, and it is the thing that actually breaks.

Why tool choice and not the answer: a wrong answer from the right tool
is a prompt problem and shows up as a bad sentence. A right answer from
the wrong tool - or from no tool at all - is the model answering from
its own weights and sounding just as confident. On a vocational course
that is the dangerous failure, because the student is examined on their
handouts, not on what a 7B model happens to remember.

Cases are grouped by which tool should win:

  calculate     arithmetic, which the model must not do in its head
  documents     "what do I have", which needs the index, not retrieval
  materials     subject questions, which must open the handouts
  summarise     "what does this file cover", one named document
  history       a signed-in student's own past sessions

There is a second evaluation below it: CALCULATOR SAFETY. It asks a
different question - can a model-written expression make the calculator
execute something - and needs no model at all, so it is kept apart and
runs in milliseconds. See evaluate_calculator_safety.

Run:  python -m tests.agent_eval
      python -m tests.agent_eval --safety-only     (no model needed)

Expected to be re-run whenever the agent model, the system prompt or a
tool description changes - all three move these numbers. Tool
descriptions move them most: they are the only thing the model sees
when choosing.
"""

import sys
import time

sys.path.insert(0, ".")

from agent.executor import run_agent
from agent.tools import calculate

# (question, the tool that must be called, user_id, group)
CASES: list[tuple[str, str, int | None, str]] = [
    # --- calculate: never in the model's head ---------------------------
    ("What is 1250 times 0.18?", "calculate", None, "calculate"),
    ("If I score 34 out of 40, what percentage is that?",
     "calculate", None, "calculate"),

    # --- documents: the index, not the vector store ----------------------
    ("What documents do I have?", "list_course_documents", None, "documents"),
    ("What material is available on data warehousing?",
     "list_course_documents", None, "documents"),

    # --- materials: must open the handouts -------------------------------
    ("What is a data warehouse?", "search_course_materials", None, "materials"),
    ("Explain ETL from my notes", "search_course_materials", None, "materials"),
    ("What is semantic HTML?", "search_course_materials", None, "materials"),
    ("What are the assessment prerequisites?",
     "search_course_materials", None, "materials"),

    # --- summarise: one named document -----------------------------------
    ("Summarise ComputerScienceOne.pdf", "summarise_document", None, "summarise"),

    # --- history: needs a signed-in user ---------------------------------
    ("What have I been studying?", "recall_my_study_history", 1, "history"),
]

# Expressions a model might produce that must never execute. Each is a
# real attempt at something: imports, attribute walking to reach a
# constructor, and an exponent that is a denial of service in three
# characters. The requirement is identical for all of them - come back
# as a sentence, having run nothing.
UNSAFE: list[tuple[str, str]] = [
    ("__import__('os').system('echo pwned')", "import"),
    ("(1).__class__.__bases__[0].__subclasses__()", "attribute walk"),
    ("open('secret.txt').read()", "file read"),
    ("2 ** 999999999", "unbounded exponent"),
    ("[1,2,3][0]", "subscripting"),
    ("print('hi')", "function call"),
]

# Expressions that must work, including the shapes a model writes when it
# has been told to show its working.
SAFE: list[tuple[str, float]] = [
    ("1250 * 0.18", 225),
    ("(34 / 40) * 100", 85),
    ("2 ** 10", 1024),
    ("17 % 5", 2),
    ("-4 + 10", 6),
    ("100 / 8", 12.5),
]


def evaluate(verbose: bool = True) -> dict:
    """Run every case and report which tool the agent actually chose."""
    rows = []

    for question, want, user_id, group in CASES:
        start = time.perf_counter()
        out = run_agent(question, user_id=user_id)
        took = time.perf_counter() - start

        used = out["tools_used"]
        rows.append({
            "question": question,
            "want": want,
            "used": used,
            "hit": want in used,
            # Chose the tool itself, vs. was made to by the backstop.
            "rescued": out.get("rescued", False),
            # What it reached for first, before any backstop. The only
            # diagnostic for why unaided choice is not improving.
            "chose_first": out.get("chose_first"),
            # The failure worth separating out: it answered a subject
            # question from its weights and never opened anything.
            "silent": not used,
            "seconds": took,
            "answer": out["answer"],
            "group": group,
        })

        if verbose:
            if not rows[-1]["hit"]:
                mark = "<--"
            elif rows[-1]["rescued"]:
                mark = "fix"      # the backstop caught it, not the model
            else:
                mark = "OK "
            got = ", ".join(used) if used else "NO TOOL"
            print(f"  {mark} want {want:24} got {got[:36]:38} "
                  f"{took:5.1f}s  {question[:34]}")

    total = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    seconds = sorted(r["seconds"] for r in rows)
    rescues = sum(1 for r in rows if r["rescued"])
    summary = {
        "total": total,
        "hits": hits,
        "accuracy": hits / total,
        # The number that says whether tool choice itself improved.
        # hits counts cases the backstop rescued; unaided does not. Kept
        # apart because a backstop that hides a flat tool-choice score
        # is worse than no backstop: the measurement stops working.
        #
        # Counted per row rather than as hits - rescues. A rescue does
        # not imply a hit: a backstop firing on a question another tool
        # should have served replaces tools_used and loses the hit. The
        # subtraction printed "-2/10" in that case; this cannot.
        "unaided": sum(1 for r in rows if r["hit"] and not r["rescued"]),
        "rescued": rescues,
        "silent": sum(1 for r in rows if r["silent"]),
        "median_seconds": seconds[total // 2],
        "slowest_seconds": seconds[-1],
        # Carried out so a repeated run can aggregate per case rather
        # than re-deriving them.
        "rows": rows,
    }

    if verbose:
        print("\n" + "=" * 62)
        print("  AGENT: does it reach for the right tool?")
        print("=" * 62)
        for group in ("calculate", "documents", "materials", "summarise", "history"):
            band = [r for r in rows if r["group"] == group]
            if band:
                print(f"  {group:12} {sum(1 for r in band if r['hit'])}/{len(band)}")
        print(f"\n  correct tool  : {hits}/{total}  ({summary['accuracy']:.0%})")
        print(f"    chose it itself : {summary['unaided']}/{total}")
        print(f"    backstop caught : {summary['rescued']}/{total}")
        # Every case the model did not get right on its own, and what
        # it picked instead. A rescue still counts as a wrong choice
        # here - the backstop saved the answer, not the decision.
        unaided_misses = [r for r in rows if r["rescued"] or not r["hit"]]
        if unaided_misses:
            print()
            print("  --- what it reached for first, where that was wrong ---")
            for row in unaided_misses:
                picked = row["chose_first"] or "nothing"
                print(f"    {picked:24} should be {row['want']:24} "
                      f"{row['question'][:30]}")

        wasted = sum(1 for r in rows if r["rescued"] and not r["hit"])
        if wasted:
            # The backstop firing where it had no business firing: it
            # cost a retrieval pass and took a correct answer away.
            print(f"    backstop MISFIRED: {wasted}/{total}")
        print(f"  no tool at all: {summary['silent']}/{total}")
        print(f"  median        : {summary['median_seconds']:.1f}s   "
              f"slowest: {summary['slowest_seconds']:.1f}s")

        # Printed in full, not counted: answering about the student's own
        # subject without reading their materials is the failure that
        # matters, and the wording is the evidence.
        silent = [r for r in rows if r["silent"]]
        print("\n  --- answered with no tool at all ---")
        for row in silent or []:
            print(f"    {row['question']}")
            print(f"      -> {row['answer'][:100]}")
        if not silent:
            print("    none")

    return summary


def evaluate_calculator_safety(verbose: bool = True) -> dict:
    """Can a model-written expression make the calculator do something?

    Scored separately and runs without a model, because it is not a
    question about judgement. The calculator is called with text the
    model wrote, so `eval` on it would be arbitrary code execution;
    this is the test that says it never happens.
    """
    blocked, computed = [], []

    for expression, kind in UNSAFE:
        result = calculate.invoke({"expression": expression})
        # The contract is a sentence back, never a value and never a
        # raise. Any result that reads like an answer means something
        # ran.
        refused = ("not a valid" in result or "Cannot work" in result
                   or "could not be calculated" in result)
        blocked.append({"expression": expression, "kind": kind,
                        "refused": refused, "result": result})

    for expression, expected in SAFE:
        result = calculate.invoke({"expression": expression})
        tail = result.split("=")[-1].strip()
        try:
            correct = abs(float(tail) - expected) < 1e-9
        except ValueError:
            correct = False
        computed.append({"expression": expression, "expected": expected,
                         "correct": correct, "result": result})

    summary = {
        "unsafe_total": len(blocked),
        "unsafe_refused": sum(1 for b in blocked if b["refused"]),
        "safe_total": len(computed),
        "safe_correct": sum(1 for c in computed if c["correct"]),
    }

    if verbose:
        print("\n" + "=" * 62)
        print("  CALCULATOR SAFETY: nothing executes")
        print("=" * 62)
        for item in blocked:
            state = "blocked" if item["refused"] else "RAN     "
            print(f"  {state} [{item['kind']:18}] {item['expression'][:38]}")
        for item in computed:
            state = "ok     " if item["correct"] else "WRONG  "
            print(f"  {state} {item['result'][:52]}")
        print(f"\n  refused      : {summary['unsafe_refused']}/{summary['unsafe_total']}")
        print(f"  computed     : {summary['safe_correct']}/{summary['safe_total']}")

    return summary


def evaluate_repeated(trials: int = 3) -> dict:
    """Run the whole set several times and report per-case stability.

    One trial per case is too thin to tune against. Unaided choice read
    7, 7, 8, 8 across single runs with code changes in between, so a
    single number cannot distinguish a real improvement from the model
    being non-deterministic. This prints the per-case hit rate, which
    separates "reliably right" from "coin flip" - the latter being the
    cases worth working on.
    """
    per_case = {question: [] for question, _, _, _ in CASES}
    summaries = []

    for trial in range(1, trials + 1):
        print()
        print("=" * 62)
        print(f"  TRIAL {trial} of {trials}")
        print("=" * 62)
        summary = evaluate(verbose=True)
        summaries.append(summary)
        for row in summary["rows"]:
            per_case[row["question"]].append(row)

    print()
    print("=" * 62)
    print(f"  ACROSS {trials} TRIALS")
    print("=" * 62)
    print(f"  {'case':34} {'correct':>9} {'unaided':>9}")
    for question, want, _, _ in CASES:
        runs = per_case[question]
        correct = sum(1 for r in runs if r["hit"])
        unaided = sum(1 for r in runs if r["hit"] and not r["rescued"])
        flag = "" if unaided == len(runs) else ("  <-- unstable"
                                                if unaided else "  <-- never")
        print(f"  {question[:34]:34} {correct:>5}/{len(runs)} {unaided:>7}/{len(runs)}{flag}")

    totals = len(CASES) * trials
    hits = sum(s["hits"] for s in summaries)
    unaided = sum(s["unaided"] for s in summaries)
    rescued = sum(s["rescued"] for s in summaries)
    silent = sum(s["silent"] for s in summaries)
    result = {
        "trials": trials,
        "cases_per_trial": len(CASES),
        "correct_rate": hits / totals,
        "unaided_rate": unaided / totals,
        "rescued_rate": rescued / totals,
        "silent_rate": silent / totals,
        "per_trial_hits": [s["hits"] for s in summaries],
        "per_trial_unaided": [s["unaided"] for s in summaries],
    }

    print()
    print(f"  correct tool : {hits}/{totals}  ({result['correct_rate']:.0%})"
          f"   per trial {result['per_trial_hits']}")
    print(f"  unaided      : {unaided}/{totals}  ({result['unaided_rate']:.0%})"
          f"   per trial {result['per_trial_unaided']}")
    print(f"  rescued      : {rescued}/{totals}  ({result['rescued_rate']:.0%})")
    print(f"  no tool used : {silent}/{totals}  ({result['silent_rate']:.0%})")
    return result


if __name__ == "__main__":
    # The safety half needs no model, so it can be run on its own - and
    # it is the half that must never regress quietly.
    if "--safety-only" in sys.argv:
        evaluate_calculator_safety()
    elif "--trials" in sys.argv:
        count = sys.argv[sys.argv.index("--trials") + 1]
        evaluate_repeated(int(count))
        evaluate_calculator_safety()
    else:
        evaluate()
        evaluate_calculator_safety()
