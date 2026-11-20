# Technical documentation

How the AI Knowledge Assistant is built, why it is built that way, and where its limits are. For setup see
the [README](../README.md); for measured results see the [test report](test_report.md).

## 1. Design principles

Three rules shaped most decisions, and they recur below.

1. **Decide in code, not in the prompt.** A local 7B model follows a formatting instruction most of the
   time, and "most of the time" is a broken quiz question or a leaked prompt. Every property that can be
   checked mechanically — where a quiz's right answer sits, whether an answer cites anything, whether a
   message is a known prompt-injection shape — is decided in code. Prompts carry only the judgement a
   model is needed for.
2. **Say where an answer came from.** Every answer is either grounded in a cited passage or explicitly
   labelled general knowledge. The interface shows the difference (a pine edge versus a grey one), and
   the evaluation measures it.
3. **Measure before changing.** Thresholds, model choices and every fix in the workflow were set from
   measurements on the evaluation sets, recorded in the code next to the decision.

## 2. Components

### 2.1 Models and routing — `llm/provider.py`
Two local Ollama models: `qwen2.5:7b` (capable) and `qwen2.5:3b` (fast). `route_llm()` routes by question
*shape*, not length — "why", "compare", "prove" go to the capable model; lookups to the fast one — and
defaults to the capable model, because misrouting a hard question costs a bad answer while misrouting an
easy one costs under a second. `safe_invoke()` turns Ollama's failure modes (server down, model not
pulled) into one `LLMUnavailableError`, surfaced as HTTP 503. Ollama is addressed as `127.0.0.1`, not
`localhost`: on the development machine `localhost` resolved to IPv6 first and every call waited out a
2-second connect timeout.

### 2.2 Ingestion — `rag/loaders.py`, `rag/splitter.py`, `rag/metadata.py`
Loaders for PDF, DOCX, Markdown, text, CSV and web pages. Splitting is token-based using the
**embedding model's own tokenizer** (character counts varied from 3 to 202 tokens for the same setting).
Each chunk gets `doc_id`, filename, type, page, `chunk_index`, ingest date, and a category/topic
assigned by the classification chain — the category is what the department/subject filter uses.

### 2.3 Vector store — `rag/vectorstore.py`, `rag/service.py`
Persistent ChromaDB with `bge-small-en-v1.5` embeddings. Re-ingesting a document deletes its old chunks
first, keyed on `doc_id`; otherwise an outdated version competes with — and can outrank — its
replacement. `VectorStoreService` wraps add / search / filter / list / get / delete, with a CLI.

### 2.4 Retrieval — `rag/retriever.py`
An absolute similarity cutoff cannot separate relevant from irrelevant on this corpus (an irrelevant
query scored *higher* than a direct hit), so retrieval is a pipeline:

1. bi-encoder search for a candidate pool, with an optional metadata filter;
2. a **cross-encoder reranker** scores each candidate against the question — this is what makes
   paraphrased questions work;
3. a **relevance gate**: nothing above the bar means *no context*, and the answer says so rather than
   grounding itself in the nearest unrelated document;
4. on a miss, **one deeper pass** with a larger pool and a stricter bar (a correct chunk ranked 18th, just
   outside the first pool).

Strategies, selectable with `RETRIEVAL_STRATEGY`: `similarity` (default), `mmr`, `multi_query`,
`compression`, and `parent` (match small chunks, return each with its neighbours as one passage). The
comparison is in [retrieval_evaluation.md](retrieval_evaluation.md).

### 2.5 Conversational RAG — `chains/tutor_chain.py`, `chains/question_rewriter.py`
A follow-up like "what about the second one?" is rewritten into a standalone question **for retrieval
only**; the model answers the student's own words, so it never answers a question nobody asked. A cheap
pre-filter skips the rewrite call for self-contained questions (0 ms median). History is stored per user
in SQLite; the last ten turns are replayed into the prompt.

### 2.6 Tool-calling agent — `agent/`
A LangGraph ReAct agent with six tools: course search, document list, document summary, calculator,
web search and the student's own study history (signed in only). It lives behind its own endpoint
because an agent loop is slower than the RAG chain and most questions do not need tools. Code-side
backstops catch three measured failures: answering a subject question without searching, crediting
"general knowledge" for a passage it did retrieve, and doing arithmetic without the calculator. The
calculator parses with `ast` and never `eval`s model output.

### 2.7 Controlled workflow — `graph/rag_workflow.py`
A LangGraph `StateGraph` for answers that must be verified before anyone sees them:

```
classify ─┬─ injection ──► refuse ─────┐
          ├─ small talk ─► small_talk ─┤
          ├─ off-topic ──► decline ────┤
          └─► search ─► generate ─► validate ─┬─ passed ──────────► finalize
                          ▲                   ├─ failed, retry ─► generate
                          └───────────────────┘└─ failed again ─► human_review ─► finalize
```

- **classify**: known injection shapes are refused **in code**, before any model; small talk is
  answered without retrieval; otherwise the model labels the message COURSE, OUT_OF_SCOPE or
  MANIPULATION. An "out of scope" label is overruled if the corpus actually answers the question.
- **search**: history-aware rewrite, subject filter, retrieval. If the rewrite finds nothing, the
  content words of this and the previous question are searched instead.
- **generate**: answers from numbered passages, citing them; passages are framed as reference material,
  never instructions.
- **validate**: with no passages, the "not covered" disclaimer is enforced and stray citations removed.
  With passages, a judge lists unsupported statements; whole flagged sentences are cut, and only if too
  little remains does the draft go back for one rewrite, then to a person.
- **human_review**: `interrupt()` pauses the run; staff approve (optionally editing) or reject in the
  Monitoring screen, which resumes the graph with `Command(resume=...)`. Runs are checkpointed in SQLite
  (`SqliteSaver`), so a pending review survives a restart.

### 2.8 Study and placement — `study/`
Quizzes, flashcards, guides and interview questions generated from passages sampled across one document.
The model writes; code decides: the correct option's position is shuffled in code, questions that refer
to "the passage" are repaired or dropped, hedged definitions are dropped, and an interview score is
computed from per-point covered/partly/missed judgements rather than asked for (asked directly, the same
answer scored 3 and 5 on consecutive runs). Quizzes are graded on the server.

### 2.9 Prompt versioning — `llm/prompt_registry.py`, `prompts.lock.json`
Workflow and evaluation prompts are registered with a name and version; the lock file records each
one's fingerprint. `python -m llm.prompt_registry --check` fails if a prompt's text changed without a
version bump. Versions are attached to every workflow result, LangSmith trace and evaluation report.

### 2.10 Observability — `monitoring.py`, `feedback.py`, Monitoring screen
Per-request timings (retrieval, model, time to first token) in a ring buffer; thumbs up/down with whether
the answer was grounded; the latest evaluation report; the review queue; the workflow graph drawn from
the compiled graph itself. LangSmith tracing is enabled by environment variables (see `.env.example`).

## 3. Data model — `database/models.py`

| Table | Holds |
|---|---|
| `users` | Google `sub` (the key — stable, unlike email), email, name, date of birth, role |
| `sessions`, `messages` | Conversations and turns, with the citations shown and the model used |
| `practice_sets` | Generated quizzes and interview questions, with answers kept server-side |
| `practice_results` | Finished practice — score out of possible — for Progress |
| `study_guides` | Cached guides, invalidated when a document's chunk count changes |
| `workflow_reviews` | The human review queue and its decisions |

SQLite with foreign keys enforced and WAL journaling, set per connection.

## 4. Security

| Concern | Measure |
|---|---|
| Who can change the knowledge base | Upload, URL ingest, delete, monitoring and review are **staff only** (`role = staff`, or a verified email in `ADMIN_EMAILS`): 401 signed out, 403 for students |
| Identity | Google OAuth authorisation-code flow, CSRF state cookie, signed HttpOnly session cookie, no passwords stored |
| Reading others' data | Every session, transcript and practice-set read is scoped to its owner |
| Faked scores | Quizzes and interviews are graded on the server; the first answer is final |
| Path traversal | Uploaded filenames are reduced to their final component |
| Server-side request forgery | URL ingest refuses any host resolving to a non-public address (loopback, private, link-local, cloud metadata) |
| Code execution | The calculator walks an AST whitelist; exponents are capped |
| Prompt injection | Known shapes refused in code; the classifier flags manipulation; passages are framed as data; answers are validated before release |
| Secrets | `.env` is gitignored and excluded from Docker images |

Known gaps: DNS rebinding between the SSRF check and the fetch; injection hidden *inside an uploaded
document* is mitigated (framing, validation, staff-only upload) but not tested end to end; there is no
rate limiting, which belongs at the reverse proxy in a public deployment.

## 5. API

| Method and path | Access | Purpose |
|---|---|---|
| `POST /ask`, `POST /ask/stream` | anyone | Conversational RAG; stream is Server-Sent Events |
| `POST /agent` | anyone | Tool-calling agent |
| `POST /workflow` | anyone | Controlled workflow |
| `GET /workflow/graph` | anyone | Workflow as Mermaid |
| `GET /workflow/result/{id}` | anyone with the id | Outcome of a held answer |
| `GET /workflow/reviews`, `POST /workflow/reviews/{id}` | staff | Review queue; approve or reject |
| `GET /documents` | anyone | What is indexed |
| `POST /upload`, `POST /ingest-url`, `DELETE /documents/{id}` | staff | Manage the knowledge base |
| `GET /study/guide/{id}`, `POST /study/quiz`, `/study/flashcards`, `/study/interview` (+ answer routes) | anyone; results saved when signed in | Practice |
| `GET /me/progress`, `/me/sessions`, `/me/stats` | signed in | The student's own data |
| `GET /stats`, `/feedback/summary`, `/eval/latest` | staff | Monitoring |
| `/auth/*` | — | Sign-in, profile, sign-out |

Interactive documentation: `http://127.0.0.1:8000/docs` when running locally.

## 6. Deployment — `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`

Four services: `ollama` (models in a volume), a one-shot `ollama-pull`, `backend` (not published), and
`frontend` (nginx serving the build and proxying `/api` to the backend, so the browser sees one origin).
The backend image installs the CPU build of torch and bakes in the embedding and reranker models, so the
container runs offline. It runs as a non-root user and writes only to the `/data` volume.

For a public deployment: serve over HTTPS with `COOKIE_SECURE=true`, set a fixed `SESSION_SECRET`, set
`PUBLIC_URL` and register `${PUBLIC_URL}/api/auth/google/callback` with Google, and enable the GPU block
in `docker-compose.yml` if a GPU is available — on CPU a streamed answer took about 90 seconds on the
development laptop (first token in under a second, so the interface stays responsive, but slowly).

The compose file pins the data paths in `environment`, which overrides `env_file`: a `.env` written for
local development sets relative paths such as `CHROMA_DB_PATH=chroma_db`, which inside the container
would fall outside the `/data` volume.

Verified end to end on Docker Desktop: all four services start in dependency order; through the nginx
proxy, signed-out upload is refused, staff upload indexes, Ask answers with citations, answers stream,
the workflow refuses an injection attempt, and every file the app writes lands in `/data`.

## 7. Limitations

- Answer quality is bounded by local 7B and 3B models. A hosted model would answer better; switching is
  confined to `llm/provider.py`, at the cost of documents and questions leaving the machine.
- The faithfulness score is an LLM judge of the same model family; per-claim results are kept in the
  report so a person can audit them.
- In-memory chat history for signed-out users resets on restart; signed-in history persists.
- The relevance gate assumes one passage should stand out; a question answered equally by several
  passages can look flat and be rejected.
