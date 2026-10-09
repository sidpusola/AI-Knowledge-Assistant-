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

There is a second evaluation below it: MULTI-TURN. It asks a different
question - given a conversation, does a follow-up retrieve the right
chunk - and is scored differently, so it is kept apart rather than
folded into the numbers above. See evaluate_multiturn.

Run:  python -m tests.retrieval_eval

Expected to be re-run whenever the embedding model, chunking or gap
threshold changes - all three move these numbers.
"""

import sys

sys.path.insert(0, ".")

from langchain_core.messages import AIMessage, HumanMessage

from chains.question_rewriter import rewrite_question
from config import settings
from rag.retriever import get_retriever, relevance_gap, retrieve, search_with_scores
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
    ("what is recursion", True, "easy"),
    ("what is a pointer", True, "easy"),
    ("what is binary search", True, "easy"),
    ("what is a stack data structure", True, "easy"),
    ("what is dynamic programming", True, "easy"),
    ("what is perception", True, "easy"),
    ("what is classical conditioning", True, "easy"),
    ("what is operant conditioning", True, "easy"),
    ("what is personality", True, "easy"),
    ("what is the Myntra internship profile", True, "easy"),
    ("what is the eligibility criteria for the Myntra internship", True, "easy"),
    ("when is the MindSprint interview", True, "easy"),

    # --- paraphrase: answerable, different words --------------------------
    ("how is data organised for business reporting", True, "paraphrase"),
    ("why should I use meaningful tags instead of div everywhere", True, "paraphrase"),
    ("what do I need to set up before my placement test", True, "paraphrase"),
    ("how long do I have to log in once the exam starts", True, "paraphrase"),
    ("what makes relationships in a household work", True, "paraphrase"),
    ("how do I move data from sources into a reporting system", True, "paraphrase"),
    ("what should my environment be like during the assessment", True, "paraphrase"),
    ("how does a function solve a problem by calling itself", True, "paraphrase"),
    ("how can I quickly find an item in a sorted list", True, "paraphrase"),
    ("how do people learn to associate one thing with another through repeated exposure", True, "paraphrase"),
    ("what traits make up who someone is", True, "paraphrase"),
    ("what software role is being offered in the campus placement drive", True, "paraphrase"),
    ("when do I need to be free for Air India's hiring process", True, "paraphrase"),
    ("how much will I be paid during the internship", True, "paraphrase"),
    # Both added after an initial substring-based "is this in the corpus"
    # check gave a false "absent" - the doc says "cognitive behavior
    # therapy" (no "-al") and "Stipend: Rs. 70,000", not the words used
    # here. Worth keeping as paraphrase cases precisely because a naive
    # keyword check would have mislabeled them, same as the system being
    # tested should not be fooled by wording either.
    ("what is cognitive behavioral therapy", True, "paraphrase"),
    ("what is the salary package for the Myntra internship", True, "paraphrase"),

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
    ("explain Freud's stages of psychosexual development", False, "adjacent"),
    ("what is a neural network", False, "adjacent"),
    ("explain a red-black tree", False, "adjacent"),
    ("what is the time complexity of merge sort in Big O notation", False, "adjacent"),
    ("what is the written test pattern for the placement drive", False, "adjacent"),

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
    """Evaluate against whatever rag.retriever.retrieve() actually does.

    This used to print only the bi-encoder gap, which is purely
    diagnostic once USE_RERANKER is on (the default) - the real
    accept/reject decision comes from the cross-encoder score at that
    point, and the gap number shown had nothing to do with why a query
    passed or failed. Fixed to compute and print whichever signal is
    actually deciding, instead of a number that looks explanatory but
    isn't.
    """
    vectorstore = load_vectorstore()
    results = []

    for query, should_retrieve, group in QUERIES:
        scored = search_with_scores(vectorstore, query, k=settings.RETRIEVER_CANDIDATES)
        gap = relevance_gap([score for _, score in scored])

        rerank_score = None
        if settings.USE_RERANKER and scored:
            from rag.reranker import rerank

            ranked = rerank(query, [doc for doc, _ in scored])
            rerank_score = ranked[0][1] if ranked else None

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
                "rerank_score": rerank_score,
            }
        )

    if verbose:
        decisive = "rerank_score" if settings.USE_RERANKER else "gap"
        for group in ("easy", "paraphrase", "adjacent", "unrelated", "chatty"):
            rows = [r for r in results if r["group"] == group]
            hits = sum(r["correct"] for r in rows)
            print(f"\n{group.upper()}  ({hits}/{len(rows)})")
            for r in rows:
                mark = "   " if r["correct"] else "XX "
                want = "retrieve" if r["expected"] else "reject  "
                value = r[decisive]
                label = "rerank" if settings.USE_RERANKER else "gap"
                shown = f"{value:8.3f}" if value is not None else "     n/a"
                print(f"  {mark}{want}  {label}={shown}  {r['query'][:46]}")

    correct = sum(r["correct"] for r in results)

    # False negatives are the costlier error for a study assistant: being
    # told "that isn't in your materials" when it is, is worse than getting
    # a general-knowledge answer to something out of scope.
    false_negatives = [r for r in results if r["expected"] and not r["got"]]
    false_positives = [r for r in results if not r["expected"] and r["got"]]

    decisive = "rerank_score" if settings.USE_RERANKER else "gap"
    answerable = [r[decisive] for r in results if r["expected"] and r[decisive] is not None]
    unanswerable = [r[decisive] for r in results if not r["expected"] and r[decisive] is not None]

    summary = {
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "false_negatives": len(false_negatives),
        "false_positives": len(false_positives),
        "decisive_metric": decisive,
        "min_answerable": min(answerable) if answerable else None,
        "max_unanswerable": max(unanswerable) if unanswerable else None,
        "margin": (min(answerable) - max(unanswerable)) if answerable and unanswerable else None,
    }

    if verbose:
        label = "rerank score" if settings.USE_RERANKER else "gap"
        threshold = settings.RERANKER_SCORE_THRESHOLD if settings.USE_RERANKER else settings.RETRIEVER_GAP_THRESHOLD
        print("\n" + "=" * 62)
        print(f"  model          : {settings.EMBEDDING_MODEL}")
        print(f"  reranker       : {settings.USE_RERANKER} ({settings.RERANKER_MODEL if settings.USE_RERANKER else 'n/a'})")
        print(f"  decisive >=    : {threshold}  ({label})")
        print(f"  accuracy       : {correct}/{len(results)}  ({summary['accuracy']:.0%})")
        print(f"  false negatives (said no, should be yes): {len(false_negatives)}")
        for r in false_negatives:
            print(f"      {label}={r[decisive]:.3f}  {r['query']}")
        print(f"  false positives (said yes, should be no): {len(false_positives)}")
        for r in false_positives:
            print(f"      {label}={r[decisive]:.3f}  {r['query']}")
        if summary["margin"] is not None:
            print(f"  weakest answerable {label} : {summary['min_answerable']:.3f}")
            print(f"  strongest noise {label}    : {summary['max_unanswerable']:.3f}")
            print(f"  MARGIN                     : {summary['margin']:+.3f}")

    return summary


# =====================================================================
#  MULTI-TURN
# =====================================================================
#
# A different question from the one above, and scored differently.
#
# The single-turn set asks "was the accept/reject decision right". This
# set asks "given what was said before, did the follow-up fetch the
# chunk it needed". A follow-up like "what about the second one?" shares
# no wording with any chunk, so as typed it either retrieves nothing or
# retrieves something irrelevant that happened to clear the gate - and
# because the answer still reads fluently, that failure is invisible
# without a test like this one.
#
# Every case runs twice, as typed and after rewriting, so the number is
# a comparison rather than a bare score. That comparison is the evidence
# that history-aware retrieval earns its extra model call.

def _turn(question: str, answer: str) -> list:
    return [HumanMessage(content=question), AIMessage(content=answer)]


# (history, follow-up, a word the correct chunk must contain, group)
CONVERSATIONS: list[tuple[list, str, str, str]] = [
    # --- pronoun: "it" / "they" stand in for the previous topic ----------
    (_turn("what is a data warehouse?",
           "A data warehouse is a central repository of integrated data "
           "used for reporting and analysis."),
     "why is it different from a database?", "warehouse", "pronoun"),
    (_turn("what is operant conditioning?",
           "Operant conditioning shapes behaviour through reinforcement "
           "and punishment."),
     "how does it use punishment?", "conditioning", "pronoun"),
    (_turn("what is OLAP?",
           "OLAP stands for Online Analytical Processing, used for "
           "multidimensional analysis of warehouse data."),
     "what is it used for?", "olap", "pronoun"),
    (_turn("what is semantic HTML?",
           "Semantic HTML uses tags such as article, nav and header that "
           "describe the meaning of the content."),
     "why does it matter for accessibility?", "semantic", "pronoun"),

    # --- ordinal: "the second one", "the first" --------------------------
    (_turn("what are the schema types in a data warehouse?",
           "The two main ones are the star schema and the snowflake schema."),
     "what about the second one?", "snowflake", "ordinal"),
    (_turn("what is ETL in data warehousing?",
           "ETL stands for Extract, Transform, Load."),
     "explain the second step", "transform", "ordinal"),
    (_turn("what are the tiers in data warehouse architecture?",
           "It is commonly described as a three tier architecture."),
     "what is in the bottom tier?", "tier", "ordinal"),

    # --- elliptical: grammatically incomplete on its own -----------------
    (_turn("what is semantic HTML?",
           "Semantic HTML uses tags such as article, nav and header."),
     "give me an example of one", "html", "elliptical"),
    (_turn("what is OLTP?",
           "OLTP stands for Online Transaction Processing, which handles "
           "day-to-day operational transactions."),
     "and how does it compare to OLAP?", "oltp", "elliptical"),
    (_turn("what are the assessment prerequisites?",
           "You need a stable internet connection and a working webcam."),
     "what about the internet speed?", "internet", "elliptical"),

    # --- topic switch: self-contained, and must NOT be dragged back to
    #     the previous subject by the rewriter ---------------------------
    (_turn("what is a data warehouse?",
           "A data warehouse is a central repository of integrated data."),
     "what is semantic HTML?", "semantic", "switch"),
    (_turn("what is semantic HTML?",
           "Semantic HTML uses tags that describe meaning."),
     "what is OLAP?", "olap", "switch"),
]


def evaluate_multiturn(verbose: bool = True) -> dict:
    """Score follow-up retrieval, as typed against rewritten."""
    retriever = get_retriever(load_vectorstore())

    def run(use_rewrite: bool) -> list[dict]:
        rows = []
        for history, followup, keyword, group in CONVERSATIONS:
            query = rewrite_question(followup, history) if use_rewrite else followup
            docs = retriever.invoke(query)
            rows.append({
                "followup": followup,
                "group": group,
                "query": query,
                "chunks": len(docs),
                "hit": any(keyword in d.page_content.lower() for d in docs),
            })
        return rows

    typed = run(False)
    rewritten = run(True)

    hits_typed = sum(r["hit"] for r in typed)
    hits_rewritten = sum(r["hit"] for r in rewritten)
    total = len(CONVERSATIONS)

    summary = {
        "total": total,
        "hits_as_typed": hits_typed,
        "hits_rewritten": hits_rewritten,
        "accuracy_as_typed": hits_typed / total,
        "accuracy_rewritten": hits_rewritten / total,
        "empty_as_typed": sum(1 for r in typed if r["chunks"] == 0),
        "empty_rewritten": sum(1 for r in rewritten if r["chunks"] == 0),
        "rewrites_made": sum(1 for r in rewritten if r["query"] != r["followup"]),
    }

    if verbose:
        print("\n" + "=" * 62)
        print("  MULTI-TURN: does a follow-up retrieve the right chunk?")
        print("=" * 62)
        for before, after in zip(typed, rewritten):
            if after["hit"] and not before["hit"]:
                change = "fixed  "
            elif before["hit"] and not after["hit"]:
                change = "BROKE  "
            else:
                change = "       "
            print(f"  {change}{'hit ' if after['hit'] else 'MISS'} "
                  f"[{after['group']:10}] {after['followup'][:32]:34}")
            if after["query"] != after["followup"]:
                print(f"            -> {after['query'][:62]}")
        print(f"\n  right chunk, as typed  : {hits_typed}/{total}  "
              f"({summary['accuracy_as_typed']:.0%})")
        print(f"  right chunk, rewritten : {hits_rewritten}/{total}  "
              f"({summary['accuracy_rewritten']:.0%})")
        print(f"  retrieved nothing      : {summary['empty_as_typed']} -> "
              f"{summary['empty_rewritten']}")
        print(f"  questions rewritten    : {summary['rewrites_made']}/{total}")

    return summary


if __name__ == "__main__":
    evaluate()
    evaluate_multiturn()
