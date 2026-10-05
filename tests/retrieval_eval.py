"""
Retrieval evaluation set.

Checks one thing: does the relevance gate accept questions this corpus can
answer and reject the ones it can't. Not answer quality - only whether
the right decision is made about passing context to the model at all.

Queries are grouped by how hard that decision is:

  easy        wording overlaps the source text directly
  paraphrase  answerable, but sharing few or no words with the source
  adjacent    NOT in the corpus, but close enough to it to be tempting -
              "data lake" against a data-warehousing corpus. The honest
              test; obvious nonsense proves very little
  unrelated   nothing to do with anything indexed
  chatty      greetings and small talk, which should never retrieve

Run:  python -m tests.retrieval_eval

Expected to be re-run whenever the embedding model, chunking or gap
threshold changes - all three move these numbers.
"""

import sys

sys.path.insert(0, ".")

from config import settings
from rag.retriever import relevance_gap, retrieve, search_with_scores
from rag.vectorstore import load_vectorstore

# (query, should_retrieve, group)
QUERIES: list[tuple[str, bool, str]] = [
    # --- easy: wording close to the source --------------------------------
    ("what is a data warehouse", True, "easy"),
    ("what is semantic HTML", True, "easy"),
    ("what is OLAP", True, "easy"),
    ("what is OLTP", True, "easy"),
    ("what is ETL", True, "easy"),
    ("three tier data warehousing architecture", True, "easy"),
    ("what are the assessment prerequisites", True, "easy"),
    ("what internet speed is required for the assessment", True, "easy"),
    ("do I need a webcam for the test", True, "easy"),
    ("harmony in the family", True, "easy"),
    ("what is affection", True, "easy"),
    ("HTML heading tags", True, "easy"),
    ("how do I make a list in HTML", True, "easy"),
    ("HTML table tags", True, "easy"),

    # --- paraphrase: answerable, different words --------------------------
    ("how is data organised for business reporting", True, "paraphrase"),
    ("why should I use meaningful tags instead of div everywhere", True, "paraphrase"),
    ("what do I need to set up before my placement test", True, "paraphrase"),
    ("how long do I have to log in once the exam starts", True, "paraphrase"),
    ("what makes relationships in a household work", True, "paraphrase"),
    ("how do I move data from sources into a reporting system", True, "paraphrase"),
    ("what should my environment be like during the assessment", True, "paraphrase"),

    # --- adjacent: tempting but NOT in the corpus -------------------------
    ("what is a data lake", False, "adjacent"),
    ("how do I normalise a database to third normal form", False, "adjacent"),
    ("what is CSS flexbox", False, "adjacent"),
    ("how do I write a JavaScript event handler", False, "adjacent"),
    ("what is a NoSQL database", False, "adjacent"),
    ("how do I resolve a conflict with my manager", False, "adjacent"),
    ("what is Kafka used for", False, "adjacent"),
    ("explain database indexing strategies", False, "adjacent"),
    ("what is responsive web design", False, "adjacent"),
    ("how do I prepare for a system design interview", False, "adjacent"),

    # --- unrelated --------------------------------------------------------
    ("what is the capital of France", False, "unrelated"),
    ("how do I bake sourdough bread", False, "unrelated"),
    ("who won the world cup", False, "unrelated"),
    ("what is the weather tomorrow", False, "unrelated"),
    ("recommend a good film", False, "unrelated"),
    ("how do I change a car tyre", False, "unrelated"),
    ("what is the population of Brazil", False, "unrelated"),

    # --- chatty -----------------------------------------------------------
    ("hello", False, "chatty"),
    ("how are you", False, "chatty"),
    ("thanks for your help", False, "chatty"),
    ("can you help me", False, "chatty"),
]


def evaluate(verbose: bool = True) -> dict:
    vectorstore = load_vectorstore()
    results = []

    for query, should_retrieve, group in QUERIES:
        scored = search_with_scores(vectorstore, query, k=settings.RETRIEVER_CANDIDATES)
        gap = relevance_gap([score for _, score in scored])
        retrieved = bool(retrieve(vectorstore, query))
        results.append(
            {
                "query": query,
                "group": group,
                "expected": should_retrieve,
                "got": retrieved,
                "correct": retrieved == should_retrieve,
                "gap": gap,
                "top": scored[0][1] if scored else 0.0,
            }
        )

    if verbose:
        for group in ("easy", "paraphrase", "adjacent", "unrelated", "chatty"):
            rows = [r for r in results if r["group"] == group]
            hits = sum(r["correct"] for r in rows)
            print(f"\n{group.upper()}  ({hits}/{len(rows)})")
            for r in rows:
                mark = "   " if r["correct"] else "XX "
                want = "retrieve" if r["expected"] else "reject  "
                print(
                    f"  {mark}{want}  gap={r['gap']:6.3f}  top={r['top']:6.3f}  {r['query'][:46]}"
                )

    correct = sum(r["correct"] for r in results)

    # False negatives are the costlier error for a study assistant: being
    # told "that isn't in your materials" when it is, is worse than getting
    # a general-knowledge answer to something out of scope.
    false_negatives = [r for r in results if r["expected"] and not r["got"]]
    false_positives = [r for r in results if not r["expected"] and r["got"]]

    answerable = [r["gap"] for r in results if r["expected"]]
    unanswerable = [r["gap"] for r in results if not r["expected"]]

    summary = {
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "min_answerable_gap": min(answerable),
        "max_unanswerable_gap": max(unanswerable),
        "margin": min(answerable) - max(unanswerable),
    }

    if verbose:
        print("\n" + "=" * 62)
        print(f"  model      : {settings.EMBEDDING_MODEL}")
        print(f"  gap >=     : {settings.RETRIEVER_GAP_THRESHOLD}")
        print(f"  accuracy   : {correct}/{len(results)}  ({summary['accuracy']:.0%})")
        print(f"  false negatives (said no, should be yes): {len(false_negatives)}")
        for r in false_negatives:
            print(f"      gap={r['gap']:.3f}  {r['query']}")
        print(f"  false positives (said yes, should be no): {len(false_positives)}")
        for r in false_positives:
            print(f"      gap={r['gap']:.3f}  {r['query']}")
        print(f"  weakest answerable gap : {summary['min_answerable_gap']:.3f}")
        print(f"  strongest noise gap    : {summary['max_unanswerable_gap']:.3f}")
        print(f"  MARGIN                 : {summary['margin']:+.3f}")

    return summary


if __name__ == "__main__":
    evaluate()
