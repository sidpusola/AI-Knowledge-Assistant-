"""
Information extraction chain - Week 2 deliverable #3.

Demonstrates RunnablePassthrough and RunnableParallel, the two LCEL
primitives the earlier chains never needed.

The problem it solves: an extraction chain's output is a structured
object, and the source text it came from is gone by the time the LLM
returns. Callers usually want both - the extracted fields *and* the text
they were pulled from, so they can show the evidence next to the claim.

The naive fix is to thread the original text back through in Python:

    extracted = chain.invoke(text)
    return {"original_text": text, "extracted": extracted}

RunnablePassthrough does it inside the chain instead, so the whole thing
stays one composable Runnable that .invoke()/.stream()/.batch() work on:

    RunnableParallel(
        original_text=RunnablePassthrough(),   # hand the input straight through
        extracted={"text": RunnablePassthrough()} | prompt | llm,
    )

Both branches receive the same input. One transforms it, one doesn't.

Note: RunnableParallel is used here for structure, not concurrency -
only one branch does real work. Step 10 (question generation) uses it for
what it's actually named after: several LLM calls running at once.
"""

from typing import Any, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field

from llm.provider import get_llm, safe_invoke

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You extract structured information from a student's study material.

Pull out only what is actually stated in the text - do not infer,
guess, or add anything from your own knowledge.

Fill in every field. Do not spend your whole answer on the first one.

key_terms: at most 5. Only genuine terms or concepts the text defines.
  The definition must be a short phrase, not a copied paragraph. If the
  text is instructions rather than teaching material, it may well define
  no terms at all - return an empty list, don't force it.
dates: any dates, years, deadlines or durations mentioned.
people_and_organisations: any named people, companies or institutions.
key_facts: 3 to 6 specific facts, each one short sentence.

Every item must be short.
""",
    ),
    ("human", "{text}"),
])


class KeyTerm(BaseModel):
    term: str = Field(description="The term being defined - a word or short phrase")
    definition: str = Field(description="Short definition, one sentence at most")


class ExtractedInfo(BaseModel):
    key_terms: list[KeyTerm] = Field(
        description="At most 5 terms the text actually defines. Empty list if it defines none."
    )
    dates: list[str] = Field(description="Dates, years, deadlines or durations mentioned")
    people_and_organisations: list[str] = Field(
        description="Named people, companies or institutions mentioned"
    )
    key_facts: list[str] = Field(
        description="3-6 specific facts, each one short sentence"
    )


def extract_info(text: str) -> dict[str, Any]:
    """Extract structured fields from text, keeping the source alongside.

    Returns {"original_text": <the input>, "extracted": ExtractedInfo}.
    """
    llm = get_llm().with_structured_output(ExtractedInfo)

    chain = RunnableParallel(
        original_text=RunnablePassthrough(),
        extracted={"text": RunnablePassthrough()} | prompt | llm,
    )

    return cast(dict[str, Any], safe_invoke(chain, text))


if __name__ == "__main__":
    from rag.loaders import load_document

    pages = load_document("uploads/cognizent tech.pdf")
    text = "\n\n".join(page.page_content for page in pages)[:3000]

    result = extract_info(text)
    info: ExtractedInfo = result["extracted"]

    print(f"(original text passed through: {len(result['original_text'])} chars)")
    print()

    print("Key terms:")
    for term in info.key_terms:
        print(f"  - {term.term}: {term.definition}")

    print("\nDates:")
    for date in info.dates:
        print(f"  - {date}")

    print("\nPeople and organisations:")
    for name in info.people_and_organisations:
        print(f"  - {name}")

    print("\nKey facts:")
    for fact in info.key_facts:
        print(f"  - {fact}")
