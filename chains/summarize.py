"""
Text summarization chain - Week 2 deliverable #1.

Demonstrates: LCEL RunnableSequence (prompt | llm) and structured output
via a Pydantic model. Reuses the shared LLM provider and error handling
built in Week 1 (get_llm, safe_invoke) rather than re-deriving them -
this and the other Week 2 utility chains are new workflows, not a new
foundation.

Uses get_llm() (the general-purpose model) rather than route_llm() -
routing in Week 1 picks a model based on how complex a *question* is,
which doesn't map cleanly onto "how long is the text to summarize".
Summarization always gets the capable model.
"""

from typing import cast

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm.provider import get_llm, safe_invoke

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a summarization assistant.

Summarize the user's text clearly and concisely, in your own words.

Also extract the key points as a short list - 3 to 5 items, each a
short standalone sentence.
""",
    ),
    ("human", "{text}"),
])


class SummaryResponse(BaseModel):
    summary: str = Field(description="A concise summary of the text, 2-4 sentences")
    key_points: list[str] = Field(
        description="3 to 5 short bullet points capturing the main ideas"
    )


def summarize_text(text: str) -> SummaryResponse:
    llm = get_llm().with_structured_output(SummaryResponse)
    chain = prompt | llm
    return cast(SummaryResponse, safe_invoke(chain, {"text": text}))


if __name__ == "__main__":
    from rag.loaders import load_document

    docs = load_document("docs/html1.md")
    full_text = "\n\n".join(doc.page_content for doc in docs)

    result = summarize_text(full_text)

    print("Summary:")
    print(result.summary)
    print()
    print("Key points:")
    for point in result.key_points:
        print(" -", point)
