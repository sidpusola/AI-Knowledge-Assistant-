from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from rag.vectorstore import load_vectorstore
from rag.retriever import get_retriever
from rag.metadata import build_citations
from llm.provider import route_llm, safe_invoke, safe_stream, strip_thinking
from monitoring import Timer, record_request

prompt=ChatPromptTemplate.from_messages([
    (
    "system",
    """
You are an AI tutor talking with a student.

FORMATTING RULE - apply this first, before writing anything:
If your answer would describe three or more things that share the same
kinds of detail (purpose, example, use case, syntax...), write a markdown
table, not a series of bolded headings. Like this:

| Tag | Purpose | Example |
|-----|---------|---------|
| `<ul>` | Unordered list | `<ul><li>Apple</li></ul>` |
| `<ol>` | Ordered list | `<ol><li>Step one</li></ol>` |

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


def _prepare_chain(question:str, category:str|None=None):
    """Retrieve context and pick a model for this question, returning a
    ready-to-run chain plus the context to feed it. Shared by both the
    non-streaming (ask_tutor) and streaming (stream_tutor) paths so
    retrieval + routing logic isn't duplicated between them.

    Retrieval still runs on the raw current question, not a history-aware
    rewritten one (e.g. "what about the second one?" won't get expanded
    into a self-contained query before hitting the retriever) - that's
    real history-aware retrieval, Week 7 scope. Only the *generation*
    step gets conversation memory here, so follow-up answers read
    naturally even though retrieval itself is still per-question.
    """
    # Timer covers getting the retriever too, not just the query - on the
    # first call the embedding model load dominates, and leaving it
    # outside the timer made "total" look unexplainably larger than
    # retrieval + llm combined.
    # A relevance floor applies (see rag/retriever.py), so this can
    # legitimately come back empty when nothing in the corpus is a good
    # match. That's intended: the prompt handles an empty context by
    # answering from general knowledge and saying so, which is better
    # than grounding in the nearest chunk of an irrelevant document.
    with Timer() as retrieval_timer:
        retriever=get_retriever(_get_cached_vectorstore(), category=category)
        docs=retriever.invoke(question)
        context=format_docs(docs)

    # llm is picked per-question - short/simple questions get the fast
    # model, longer/complex ones get the general-purpose model. See
    # llm/provider.py:route_llm for the routing logic.
    llm=route_llm(question)

    chain=prompt|llm|StrOutputParser()
    return chain, context, docs, llm.model, retrieval_timer.seconds


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
            chain, context, docs, model, retrieval_seconds = _prepare_chain(question, category)

            with Timer() as llm_timer:
                answer=safe_invoke(chain, {
                    "context":context,
                    "question":question,
                    "history":history
                })
            llm_seconds=llm_timer.seconds

            # Citations come from the chunks that actually went into the
            # prompt (post-retrieval-gate, post-reranking) - not every
            # candidate the retriever looked at, which the model never saw
            # and would be a misleading citation.
            return {
                "answer":strip_thinking(answer),
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
        chain, context, docs, model, retrieval_seconds = _prepare_chain(question, category)

        if sources is not None:
            sources.extend(build_citations(docs))

        chunks=safe_stream(chain, {
            "context":context,
            "question":question,
            "history":history or []
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
