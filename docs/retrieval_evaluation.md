# Retrieval evaluation — which retriever produces better answers?

Week 6 asks for a comparison of retriever strategies and a judgement on which produces better answers.
This report covers the comparison method, the results, and the decision taken.

**Short answer:** the cross-encoder reranker is the single change that matters — without it the right
document is found a third as often. Among reranked strategies, **parent-document retrieval produced the
best answers** (11/12 correct, 4/4 honest about gaps, every claim supported) at no retrieval cost, but its
lead over the default rests on one question, so the default stays `similarity` until a larger set
confirms it. Contextual compression and multi-query cost several times more time without better answers.

## Method

`python -m tests.retriever_comparison` (raw numbers: `reports/retriever_comparison.md`).

- **Questions:** the 16 answerable, follow-up and unanswerable cases from `tests/data/eval_dataset.json`
  (the same set as the Week 11 evaluation). Follow-ups are rewritten once, up front, so every strategy
  searches the same standalone question.
- **Same everything else:** one answer prompt (`workflow.answer@v1`), one model (`qwen2.5:7b`); only the
  retrieved passages differ.
- **Measured per strategy**
  - *Right document* — the expected document is among the passages (12 grounded questions)
  - *Facts correct* — the expected facts appear in the answer
  - *Faithfulness* — share of the answer's claims the passages support, judged claim by claim
    (`eval.faithfulness@v1`)
  - *Rejects unanswerable* — nothing retrieved for the 4 questions the corpus does not answer
  - *Says "not covered"* — the answer to those 4 admits it
  - context size and median retrieval / answer time

## Strategies

| Strategy | What it does |
|---|---|
| `threshold` | Bi-encoder similarity with a score floor and a gap test — no reranker (the Week 4 approach) |
| `similarity` | Similarity shortlist, cross-encoder rerank, relevance gate, deeper pass on a miss (**default**) |
| `mmr` | Maximal marginal relevance for diversity, then rerank |
| `multi_query` | The model writes rephrasings; results are pooled, then reranked |
| `compression` | Rerank shortlist, then the model extracts only the relevant part of each passage |
| `parent` | `similarity`, then each matching chunk is widened to its neighbouring chunks |

## Results

| Strategy | Right doc | Facts correct | Faithfulness | Rejects unanswerable | Says "not covered" | Context (chars) | Retrieval p50 | Answer p50 |
|---|---|---|---|---|---|---|---|---|
| threshold | 33% (4/12) | 75% | 100% | 100% | 75% | 464 | 0.03s | 0.71s |
| similarity | 83% (10/12) | 83% | 100% | 100% | 100% | 1109 | 0.17s | 0.80s |
| mmr | 83% | 92% | 100% | 100% | 75% | 942 | 0.14s | 0.81s |
| multi_query | 92% (11/12) | 75% | 100% | 100% | 100% | 1274 | 0.81s | 0.85s |
| compression | 83% | 83% | 97% | 100% | 75% | 385 | 2.31s | 0.79s |
| **parent** | 83% | **92%** | **100%** | 100% | **100%** | 2213 | 0.15s | 0.93s |

### Per question

`doc` = right document retrieved, `---` = not; `ok` / `xx` = expected facts present or missing.
For unanswerable questions: `rej` = nothing retrieved; `ok` / `xx` = the answer said "not covered".

| Case | threshold | similarity | mmr | multi_query | compression | parent |
|---|---|---|---|---|---|---|
| ans-01 internet speed | ---/xx | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| ans-02 F5 / refresh | ---/ok | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| ans-03 OLTP | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| ans-04 ETL | ---/ok | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| ans-05 Myntra eligibility | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| ans-06 affection | ---/xx | doc/ok | doc/ok | doc/ok | **doc/xx** | doc/ok |
| ans-07 semantic HTML | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| ans-08 Myntra profile | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| fol-01 vs operational DB | ---/ok | doc/ok | doc/ok | doc/ok | doc/ok | doc/ok |
| fol-02 internet speed (follow-up) | ---/xx | ---/xx | ---/xx | ---/xx | ---/xx | ---/xx |
| fol-03 OLTP vs OLAP | ---/ok | **doc/xx** | doc/ok | doc/xx | doc/ok | doc/ok |
| fol-04 second ETL step | ---/ok | ---/ok | ---/ok | **doc**/xx | ---/ok | ---/ok |
| una-01 data lake | rej/ok | rej/ok | rej/ok | rej/ok | rej/ok | rej/ok |
| una-02 CSS flexbox | rej/ok | rej/ok | rej/ok | rej/ok | rej/ok | rej/ok |
| una-03 Kafka | rej/ok | rej/ok | rej/ok | rej/ok | rej/ok | rej/ok |
| una-04 responsive design | rej/xx | rej/ok | rej/xx | rej/ok | rej/xx | rej/ok |

## What the results show

1. **The reranker is the decisive component.** `threshold` finds the right document for only 4 of 12
   questions; every reranked strategy finds 10 or 11. The bi-encoder gap test rejects paraphrased
   questions and follow-ups that a cross-encoder recognises. Its "facts correct" of 75% is misleading:
   most of those answers came from the model's general knowledge with no document behind them — exactly
   what the grounding requirement exists to prevent.

2. **Some differences are noise, and the table shows which.** On una-04, *every* strategy retrieved
   nothing, so the answer prompt received identical input — yet three of the six answers left out the
   "not covered" sentence. That column varies with the model's sampling, not with retrieval. With 12
   grounded questions, one question is 8 percentage points, so single-question gaps should not decide
   anything on their own.

3. **Contextual compression loses content.** Its extraction step dropped the definition in ans-06,
   lowered faithfulness slightly (97%), and made retrieval 14× slower (2.31s), because it calls the
   model once per candidate passage. The smaller context does not make answers faster.

4. **Multi-query finds more but answers no better.** It retrieved the right document for one extra
   follow-up (fol-04) but produced fewer correct answers (75%) and is 5× slower to retrieve, because it
   first asks the model for rephrasings.

5. **Parent-document retrieval gave the best answers**: correct on 11 of 12, honest on all 4
   unanswerable questions, every claim supported — at the same retrieval speed as `similarity`. The wider
   passage (about twice the context) carried the comparison that `similarity`'s narrower chunk missed on
   fol-03. The cost is about 0.13s more answer time from the larger prompt.

6. **One question defeats every strategy.** fol-02 ("What about the internet speed?" after a question on
   assessment prerequisites) is a question-rewriting failure, not a retrieval one: the rewrite asks about
   "the course" and loses "assessment". The controlled workflow handles it with a keyword fallback
   (see the test report); this comparison deliberately uses the plain rewrite for all strategies.

## Decision

- **Keep the reranker in every configuration.** It is the difference between 33% and 83%.
- **Default stays `similarity`.** `parent` is the best candidate, but its margin is one question.
  Changing the default on that would be fitting the test set; it should be confirmed on a larger
  question set first. Switching is one setting: `RETRIEVAL_STRATEGY=parent`.
- **Do not use `compression` or `multi_query` by default** on a local model: the extra model calls cost
  more than they return.

## Limitations

- 16 questions over one corpus. Enough to separate the reranker from the rest and to expose the costs,
  not to rank strategies that differ by a single answer.
- Faithfulness is judged by a model of the same family as the one answering. This comparison keeps
  one score per answer (`reports/retriever_comparison.json`); the workflow evaluation keeps the
  per-claim judgements (`reports/eval_report.json`) for audit.
- Timings are medians on a laptop with Ollama running locally; absolute values will differ elsewhere,
  ratios less so.
