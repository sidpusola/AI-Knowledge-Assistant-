"""
Question generation chain - Week 2 deliverable #4.

Demonstrates RunnableParallel used for actual concurrency, which is the
difference from extract.py. There, RunnableParallel held one LLM branch
next to a passthrough - useful structurally, but only one branch did real
work. Here all three branches are separate LLM calls, each with its own
prompt and its own view of what makes a good question:

    RunnableParallel(
        factual=...    | llm,   |
        conceptual=... | llm,   |  same input, three different prompts
        applied=...    | llm,   |
    )

LangChain runs sync branches in a thread pool, so they are dispatched at
the same time rather than one after another. Whether that saves any
wall-clock time depends on the backend, and here it mostly doesn't:
measured on this machine, parallel beat sequential by only 1.07-1.19x,
not the 3x the shape suggests. A local Ollama serves roughly one request
per model at a time, so the three calls queue behind each other however
they were dispatched - the concurrency is real in LangChain and absent in
the backend.

Worth knowing before reaching for RunnableParallel expecting a speedup.
Against a hosted API that accepts concurrent requests the picture would
be very different; against one local model it buys little. The __main__
block re-runs that comparison, with a warm-up first - without one, the
first timing is dominated by model load and the result is meaningless.

Three question types, because asking the same thing three ways is the
point - a student who can recall a definition may still not be able to
apply it.
"""

from typing import cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field

from llm.provider import get_llm, safe_invoke


class Question(BaseModel):
    question: str = Field(description="The question to ask the student")
    answer: str = Field(description="The expected answer, in one or two sentences")


class QuestionSet(BaseModel):
    factual: Question
    conceptual: Question
    applied: Question


def _build_prompt(instruction: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            f"""
You write practice questions for a student, based only on the material
they give you. Do not ask about anything the material does not cover.

{instruction}

Give the question and the expected answer. Keep both short.
""",
        ),
        ("human", "{text}"),
    ])


factual_prompt = _build_prompt(
    "Write one FACTUAL question - it should test whether the student can "
    "recall a specific fact, term, or definition stated in the material."
)

conceptual_prompt = _build_prompt(
    "Write one CONCEPTUAL question - it should test whether the student "
    "understands why something works or how two ideas relate, not just "
    "whether they memorised it."
)

applied_prompt = _build_prompt(
    "Write one APPLIED question - it should test whether the student can "
    "use the idea in a new situation that is not described in the material."
)


def generate_questions(text: str) -> QuestionSet:
    """Generate a factual, conceptual and applied question from material."""
    llm = get_llm()

    chain = RunnableParallel(
        factual=factual_prompt | llm.with_structured_output(Question),
        conceptual=conceptual_prompt | llm.with_structured_output(Question),
        applied=applied_prompt | llm.with_structured_output(Question),
    )

    result = cast(dict, safe_invoke(chain, {"text": text}))
    return QuestionSet(**result)


if __name__ == "__main__":
    from time import perf_counter

    from rag.loaders import load_document

    pages = load_document("docs/html1.md")
    material = "\n\n".join(page.page_content for page in pages)[:2500]

    questions = generate_questions(material)

    for label, q in [
        ("Factual", questions.factual),
        ("Conceptual", questions.conceptual),
        ("Applied", questions.applied),
    ]:
        print(f"{label}: {q.question}")
        print(f"  -> {q.answer}")
        print()

    # Is the parallelism real? Compare against running the same three
    # calls one after another.
    #
    # The warm-up below matters: whichever version runs first otherwise
    # absorbs the model load and looks far slower than it is. Timing them
    # cold once gave "parallel is 2.5x slower", which was an artifact
    # entirely - with the warm-up it's a modest win instead.
    llm = get_llm().with_structured_output(Question)
    prompts = (factual_prompt, conceptual_prompt, applied_prompt)

    def run_sequential():
        for p in prompts:
            safe_invoke(p | llm, {"text": material})

    run_sequential()  # warm-up, not measured

    start = perf_counter()
    run_sequential()
    sequential_seconds = perf_counter() - start

    start = perf_counter()
    generate_questions(material)
    parallel_seconds = perf_counter() - start

    print(f"sequential: {sequential_seconds:.2f}s")
    print(f"parallel:   {parallel_seconds:.2f}s")
    print(f"speedup:    {sequential_seconds / parallel_seconds:.2f}x")
