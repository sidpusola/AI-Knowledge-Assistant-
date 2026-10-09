import logging
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from rag.vectorstore import load_vectorstore
from rag.retriever import get_retriever
from rag.metadata import build_citations
from llm.provider import route_llm, safe_invoke, safe_stream, strip_thinking
from monitoring import Timer, record_request
from diagrams import log_answer_diagrams
from chains.diagram_planner import plan_diagram, plan_directive
from chains.question_rewriter import rewrite_question

logger = logging.getLogger(__name__)

prompt=ChatPromptTemplate.from_messages([
    (
    "system",
    """
You are an AI tutor talking with a student.

WHAT YOU CANNOT DO: you write text and draw diagrams of processes. You
cannot produce pictures, photographs or artwork of things - no image of
a dog, a cell under a microscope, or a painting. If a student asks for
one, say in a single sentence that you can't make pictures, and offer
what you can do instead (describe it, or diagram how something works).
Never answer such a request with a diagram of a different subject.

ANSWER THE QUESTION ACTUALLY ASKED. Earlier turns are there for
follow-ups like "what about the second one?". When a new question is
about something else, drop the previous topic completely - do not
reinterpret the question so it fits what you were just discussing.

FORMATTING RULE - apply this first, before writing anything:
If your answer would describe three or more things that share the same
kinds of detail (purpose, example, use case, syntax...), write a markdown
table, not a series of bolded headings. Like this:

| Tag | Purpose | Example |
|-----|---------|---------|
| `<ul>` | Unordered list | `<ul><li>Apple</li></ul>` |
| `<ol>` | Ordered list | `<ol><li>Step one</li></ol>` |

{diagram_plan}

YOU ARE ALSO A DIAGRAM PLANNER. Before drawing anything, decide what
KIND of thing the answer describes. Do not default to a linear
flowchart - most subjects are not one.

  cycle          repeats forever, the last stage leads back to the
                 first. Life cycles, the water cycle, feedback loops,
                 the menstrual cycle.
  timeline       events in time order that do not repeat. History,
                 project phases, the stages of an era.
  linear         one step causes the next and it ends. An algorithm,
                 a recipe, an installation procedure.
  decision_tree  conditional logic, yes/no routes, troubleshooting.
  hierarchy      parent and child, classification. "Types of X."
  parallel       work that splits, runs at once, and rejoins.

Then write it in the matching format:

  cycle, timeline, linear          -> ```flow-json  (JSON, below)
  decision_tree, hierarchy, parallel -> ```mermaid   (FORMAT (b))

FORMAT (a), structured JSON. Exactly this shape, nothing else:

```flow-json
{{
  "diagram_type": "cycle",
  "title": "Life cycle of a butterfly",
  "theme": "biology",
  "nodes": [
    {{"id": "egg", "label": "Egg", "description": "Laid on a host leaf", "icon": "circle-dot"}},
    {{"id": "larva", "label": "Caterpillar", "description": "Feeds and grows", "icon": "worm"}},
    {{"id": "pupa", "label": "Chrysalis", "description": "Metamorphosis", "icon": "package"}},
    {{"id": "adult", "label": "Butterfly", "description": "Reproduces", "icon": "bug"}}
  ],
  "edges": [
    {{"source": "egg", "target": "larva"}},
    {{"source": "larva", "target": "pupa"}},
    {{"source": "pupa", "target": "adult"}},
    {{"source": "adult", "target": "egg"}}
  ]
}}
```

Rules for the JSON:
- Valid JSON only. No comments, no trailing commas, nothing inside the
  block but the object.
- "diagram_type" is one of: cycle, timeline, linear.
- "theme" sets the colour treatment. Use "biology" for living things,
  "technical" for computing and engineering, "education" for study and
  learning topics, "minimal" for anything else.
- NO artificial "Start" or "End" nodes. A node must be a real stage of
  the subject. "Menstruation" is a stage; "Start" is not. This is the
  single most common mistake - a diagram that opens with a box labelled
  Start is wrong.
- "id" is short, starts with a letter, letters/digits/hyphen/underscore
  only, and appears once.
- "label" is the stage's real name, two or three words.
  "description" is one short clause saying what happens there, and may
  be left out.
- "icon" must come from the list below, or be left out. Anything not on
  the list is ignored:
  book-open, box, brain, bug, calculator, calendar, check
  check-circle-2, circle-dot, clock, cloud, cloud-rain, code, cog, cpu
  database, download, droplet, egg, file-text, filter, flag, flask-conical
  flower-2, git-branch, globe, hard-drive, key, layers, leaf, lightbulb
  list-checks, lock, mail, microscope, monitor, network, package, pen-line
  play, repeat, refresh-cw, rocket, search, send, server, settings
  shield, smartphone, sprout, sun, table, target, terminal, thermometer
  triangle-alert, upload, users, wind, worm, x, x-circle, zap
- Pick an icon that depicts that particular stage. If nothing in the
  list fits, leave "icon" out - an absent icon is fine, but giving
  every stage the same icon looks broken.
- Edges run in order from each stage to the next, using "source" and
  "target". For a cycle, the last edge returns to the first stage -
  that is what makes it a cycle.
- At most 12 stages.
- Optional "role" on a node: "decision" marks a branch point. Leave it
  out otherwise; the first and last stage are styled automatically.

FORMAT (b), Mermaid - for decision_tree, hierarchy and parallel ONLY.

These three are graphs: they branch, fan out, or split and rejoin, and
they need a layout engine. Everything that is one chain of stages - a
cycle, a timeline, a straight procedure - belongs in FORMAT (a) above,
even when the question says the word "flowchart". Do not reach for
Mermaid just because the student asked for a flowchart.

Use a diagram only when a picture genuinely helps; a definition or a
comparison does not need one.

Before drawing, find the STRUCTURE. A flowchart where every box has one
arrow in and one arrow out is a numbered list wearing a costume - it
shows the student nothing the sentences didn't already say. So ask:
- Where does it branch? Almost every real process has a decision, a
  check, an alternative route. Draw those as {{"..."}} diamonds with
  labelled edges.
- What happens when it fails? Error paths, retries and loops back to an
  earlier step are usually the most informative part of the picture.
- Which steps belong together? Group them inside a `subgraph` so the
  student sees three named phases instead of twelve equal boxes.
- Does anything run in parallel and rejoin later? Show the split and the
  merge.

Find the structure that is really there - never invent a branch to make
the picture look busier. Two hard rules:
- A diamond is ONLY for a point where the flow splits on a yes/no test,
  and it must have at least two labelled arrows leaving it. A stage in a
  sequence is a rectangle. Drawing "Pupate" or "Deploy" as a diamond is
  wrong: nothing is being decided there.
- If the subject is a cycle (a lifecycle, a feedback loop, a retry),
  close the loop back to the first step and say so with a labelled edge.

If after all that the process genuinely is a straight line - some are,
like a lifecycle or a compiler pipeline - that's fine. An honest
straight line beats a fake branch.

DIRECTION - decide this before you write the first line:
- A sequence with no branches, or only one: `flowchart LR`. It runs
  across the page and fits on screen. The same steps as `flowchart TD`
  make a tall narrow ribbon the reader has to scroll past.
- Two or more branch points, or a tree that fans out: `flowchart TD`.
  Branching needs vertical room to stay untangled.
A six-stage lifecycle is LR. A decision tree is TD.

Keep labels to two or three words where you can ("Load kernel", not
"The bootloader loads the kernel into memory"). Long labels make wide
boxes and an unreadable diagram; the sentences around it carry the
detail.

Syntax example - note the branch, the loop back and the grouping:

```mermaid
flowchart TD
    A(["Start"]) --> B["Collect input"]
    B --> C{{"Valid?"}}
    C -->|No| D["Show error"]
    D --> B
    subgraph Storage
        E["Write record"]
        F["Update index"]
    end
    C -->|Yes| E
    E --> F
    F --> G{{"More items?"}}
    G -->|Yes| B
    G -->|No| H(["Done"])
```

That example exists ONLY to show syntax and shape. It is not about any
real topic and must never appear in an answer. If a question doesn't
call for a diagram, write no diagram at all - do not reach for the
example to fill space. A definition ("what is X") almost never needs one.

Mermaid rules, follow them exactly or the diagram will not draw:
- Start with `flowchart LR` (left-right) or `flowchart TD` (top-down),
  chosen by the DIRECTION rule above. Use `sequenceDiagram` for messages
  between parties over time.
- Always wrap label text in double quotes: `A["Parse the input"]`. This
  is what keeps brackets, commas and apostrophes from breaking the parse.
- Node ids are short and plain: A, B, C1, Step2. Never put spaces or
  punctuation in an id - the label carries the words.
- `-->` is a plain arrow, `-->|Yes|` labels it, `{{"..."}}` is a decision
  diamond, `["..."]` a step, `(["..."])` a start or end.
- Group related steps with `subgraph Name` ... `end`. Declare each node
  once; put an arrow that crosses a group boundary outside the subgraph.
- Keep it under about 14 nodes. A diagram nobody can read helps nobody.
- Never write `style`, `classDef`, `class`, `fill` or any colour. The
  app colours the diagram by role. Attempts to style it here produce
  invalid syntax and the diagram is lost.
- No "Start" or "End" boxes unless the process genuinely has a distinct
  entry or exit worth naming.
- For messages between two parties over time - a handshake, a request
  and its response - use `sequenceDiagram`, not a flowchart.

Write a sentence or two around the diagram explaining it. The diagram
supports the explanation; it does not replace it.

Everything else: write plain conversational paragraphs, like a person
explaining something to a friend. No emoji as section markers, no bolded
mini-heading over every paragraph, no hype ("this is critical", "let's
dive in"), and never invent statistics or percentages. Use **bold** for a
few key terms, not every other phrase. Bullet lists only when the content
really is a list of parallel items.

Greetings and small talk ("how are you", "thanks"): just reply naturally
and warmly. Don't mention the context or course materials at all.

For real questions, ground your answer in the provided context - it's the
student's own course materials. If the context doesn't cover the
question, say so in one short sentence, then answer from general
knowledge. Say it once; don't repeat it at the end or add a paragraph
about how the question relates to the materials.

Match the length of your answer to what the question actually needs: a
couple of sentences for a simple definition, more when the question is
genuinely involved. Length should come from substance, never from
padding - don't restate the question, don't preface with what you're
about to do, don't walk through every detail in the context, and don't
end with a summary of what you just said.

Use the conversation so far to understand follow-ups ("what about the
second one?").

context:
{context}
"""
    ),
    MessagesPlaceholder("history"),
    (
       "human",
       "{question}"
    )
])

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


_vectorstore=None

def _get_cached_vectorstore():
    """Load the vectorstore once per process instead of per request.

    load_vectorstore() reconstructs the HuggingFace embedding model every
    time it's called, which measured at ~8-10s per request - roughly 40%
    of total latency, entirely wasted repeat work. Chroma reads through to
    the same persisted directory, so a cached handle still sees documents
    uploaded later (verified by uploading and immediately querying).

    The *vectorstore* is cached rather than the retriever, because each
    request may want a different filter and as_retriever() is a cheap
    wrapper - it's load_vectorstore() that costs.
    """
    global _vectorstore

    if _vectorstore is None:
        _vectorstore=load_vectorstore()

    return _vectorstore


def _prepare_chain(question:str, history:list|None=None, category:str|None=None):
    """Retrieve context and pick a model for this question, returning a
    ready-to-run chain plus the context to feed it. Shared by both the
    non-streaming (ask_tutor) and streaming (stream_tutor) paths so
    retrieval + routing logic isn't duplicated between them.

    Retrieval is history-aware: a follow-up is first rewritten into a
    question that stands on its own, and *that* is what gets embedded.
    "What about the second one?" retrieves nothing as typed - it shares
    no wording with any chunk - so before this the follow-up silently
    lost its grounding while the answer still read fluently.

    Two different strings come out of this, deliberately:
      search_query  what the retriever, router and diagram planner see,
                    rewritten to be self-contained
      question      what the student typed, which is what the generator
                    answers and what the citations belong to

    Mixing them up would mean answering a question nobody asked."""
    # Timer covers getting the retriever too, not just the query - on the
    # first call the embedding model load dominates, and leaving it
    # outside the timer made "total" look unexplainably larger than
    # retrieval + llm combined.
    # A relevance floor applies (see rag/retriever.py), so this can
    # legitimately come back empty when nothing in the corpus is a good
    # match. That's intended: the prompt handles an empty context by
    # answering from general knowledge and saying so, which is better
    # than grounding in the nearest chunk of an irrelevant document.
    # Rewriting is timed inside retrieval because that is what it is
    # for - the cost belongs to the retrieval step, not to generation.
    # Self-contained questions skip the model entirely and this is ~0ms;
    # see chains/question_rewriter.py.
    with Timer() as retrieval_timer:
        search_query=rewrite_question(question, history)
        retriever=get_retriever(_get_cached_vectorstore(), category=category)
        docs=retriever.invoke(search_query)
        context=format_docs(docs)

    if search_query != question:
        logger.info("[rewrite] %r -> %r", question, search_query)

    # llm is picked per-question - short/simple questions get the fast
    # model, longer/complex ones get the general-purpose model. See
    # llm/provider.py:route_llm for the routing logic.
    llm=route_llm(search_query)

    # One extra short call on the fast model, and only when the question
    # actually asks for a picture. Asking the tutor model to choose the
    # format inside its own prompt measured 5/9; deciding it here with a
    # constrained output measured 11/11. See chains/diagram_planner.py.
    diagram_plan=plan_directive(plan_diagram(search_query))

    chain=prompt|llm|StrOutputParser()
    return chain, context, docs, llm.model, retrieval_timer.seconds, diagram_plan


def get_tutor_chain():
    def answer_question(inputs:dict) -> dict:
        question=inputs["question"]
        history=inputs.get("history", [])
        category=inputs.get("category")

        started=perf_counter()
        model="unknown"
        retrieval_seconds=0.0
        llm_seconds=0.0
        error=None

        try:
            chain, context, docs, model, retrieval_seconds, diagram_plan = _prepare_chain(
                question, history, category
            )

            with Timer() as llm_timer:
                answer=safe_invoke(chain, {
                    "context":context,
                    "question":question,
                    "history":history,
                    "diagram_plan":diagram_plan
                })
            llm_seconds=llm_timer.seconds

            # Citations come from the chunks that actually went into the
            # prompt (post-retrieval-gate, post-reranking) - not every
            # candidate the retriever looked at, which the model never saw
            # and would be a misleading citation.
            cleaned=strip_thinking(answer)

            # Logs (never raises) when the model emits a diagram the
            # renderer would reject. The browser validates independently;
            # this is the server's own count of how often that happens.
            log_answer_diagrams(cleaned)

            return {
                "answer":cleaned,
                "sources":build_citations(docs),
            }
        except Exception as e:
            error=type(e).__name__
            raise
        finally:
            # Recorded on both the success and failure paths, so a failed
            # request still shows up in /stats instead of vanishing.
            record_request(
                question=question,
                model=model,
                retrieval_seconds=retrieval_seconds,
                llm_seconds=llm_seconds,
                total_seconds=perf_counter()-started,
                error=error,
            )

    return RunnableLambda(answer_question)

def ask_tutor(question:str, history:list|None=None, category:str|None=None) -> dict:
     """Returns {"answer": str, "sources": list[str]}.

     sources is a list of human-readable citations ("notes.pdf, page 3"),
     one per distinct chunk actually used - empty when nothing cleared
     the relevance gate and the answer came from general knowledge.
     """
     chain=get_tutor_chain()
     return chain.invoke({
         "question":question,
         "history":history or [],
         "category":category
     })


def stream_tutor(
    question:str,
    history:list|None=None,
    category:str|None=None,
    sources:list|None=None,
):
    """Yield the answer incrementally, chunk by chunk, instead of
    building the full string before returning anything - powers the
    /ask/stream endpoint so the UI can render tokens as they arrive.

    Citations are known as soon as retrieval finishes, before the first
    token is generated - but this is a generator that yields only text
    chunks (the thing /ask/stream turns straight into SSE token events),
    so there's no return value to carry them in. If a `sources` list is
    passed in, it's populated as a side effect - the caller reads it back
    after the generator is exhausted.
    """
    started=perf_counter()
    model="unknown"
    retrieval_seconds=0.0
    time_to_first_token=None
    error=None

    try:
        chain, context, docs, model, retrieval_seconds, diagram_plan = _prepare_chain(
            question, history, category
        )

        if sources is not None:
            sources.extend(build_citations(docs))

        chunks=safe_stream(chain, {
            "context":context,
            "question":question,
            "history":history or [],
            "diagram_plan":diagram_plan
        })

        for chunk in chunks:
            if time_to_first_token is None:
                # What the user actually perceives as "it started answering" -
                # the metric that justifies having built streaming at all.
                # Note this is time from the start of generation, after the
                # leaked reasoning trace (if any) has been stripped.
                time_to_first_token=perf_counter()-started
            yield chunk
    except Exception as e:
        error=type(e).__name__
        raise
    finally:
        total_seconds=perf_counter()-started
        record_request(
            question=question,
            model=model,
            retrieval_seconds=retrieval_seconds,
            llm_seconds=total_seconds-retrieval_seconds,
            total_seconds=total_seconds,
            streamed=True,
            time_to_first_token=time_to_first_token,
            error=error,
        )


if __name__=="__main__":
      result=ask_tutor(
         "what is semantic HTML?"
      )
      print(result["answer"])
      print()
      print("Sources:", result["sources"] or "(none - general knowledge)")
