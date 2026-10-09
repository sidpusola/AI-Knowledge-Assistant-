"""
Contextual compression - ask the LLM to pull only the sentences relevant
to the question out of a retrieved chunk, instead of passing the whole
chunk through (or rejecting it whole).

This is aimed squarely at what step 22 found: a 644-character chunk that
mixes leftover Forms markup, the Semantic HTML section, Best Practices
and the start of Summary. The two sentences that actually answer "why
use meaningful tags instead of div" are in there, but they're a fifth of
the chunk - the cross-encoder judges the whole thing and scores it below
threshold because most of it isn't on topic. No rephrasing of the
question changes what's inside the chunk, which is why multi-query
didn't recover it either.

Compression extracts before judging, rather than judging the whole
chunk: if the extraction comes back non-empty, that chunk had something
relevant in it, however diluted. See the __main__ block, which runs this
on exactly that chunk and checks whether it works, rather than assuming.

Cost: one LLM call per candidate chunk. Against the full RETRIEVER_CANDIDATES
(10) that's an order of magnitude more calls than reranking (one
cross-encoder pass over all 10 at once, ~85ms total) or multi-query (3-4
calls total) - measured ~4-8s per query. rag/retriever.py now runs the
same cheap cross-encoder pass first and only compresses the top
COMPRESSION_CANDIDATES (4 by default), cutting that to ~1.3s with no
measured change in outcome - the chunk this module was built for ranks
3rd of 10 by cross-encoder score even though it fails the absolute
threshold, so a top-4 shortlist still reaches it.

Separately: repeated testing shows extract_relevant() does not reliably
recover that chunk - 5/5 trials rejected it outright even though it
contains "Semantic HTML improves readability and accessibility" and
"Semantic elements help search engines understand webpage structure",
both arguably relevant. The prefilter above makes compression cheaper;
it does not make this particular case work. Treat that as open, not
fixed.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from llm.provider import get_llm, safe_invoke

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
A student asked a question. Below is one chunk of their course material,
which may only partly relate to it - it could mix content from several
sections.

Pull out ONLY the sentences that actually help answer the question,
copied exactly as they appear - do not paraphrase, summarise or add
anything. If nothing in the chunk is relevant, say so.

chunk:
{chunk}
""",
    ),
    ("human", "{question}"),
])


class Extraction(BaseModel):
    has_relevant_content: bool = Field(
        description="Whether any part of the chunk actually helps answer the question"
    )
    excerpt: str = Field(
        description="The relevant sentences, copied exactly. Empty string if none."
    )


def extract_relevant(question: str, chunk: Document) -> Document | None:
    """Return a copy of the chunk with only its relevant text, or None.

    None means the chunk had nothing worth keeping - the compression
    equivalent of being rejected by the relevance gate.
    """
    llm = get_llm().with_structured_output(Extraction)
    chain = _prompt | llm

    result = safe_invoke(chain, {"question": question, "chunk": chunk.page_content})

    if not result.has_relevant_content or not result.excerpt.strip():
        return None

    return Document(page_content=result.excerpt.strip(), metadata=chunk.metadata)


def compress_candidates(question: str, candidates: list[Document]) -> list[Document]:
    """Extract from every candidate, dropping the ones with nothing relevant.

    One LLM call per candidate - see the module docstring for the cost
    this carries relative to reranking or multi-query.
    """
    compressed = []
    for chunk in candidates:
        result = extract_relevant(question, chunk)
        if result is not None:
            compressed.append(result)
    return compressed


if __name__ == "__main__":
    import time

    # The exact chunk step 22 found: four sections mixed together, the
    # cross-encoder scored it -8.889 against a -3.5 threshold.
    messy_chunk = Document(
        page_content=(
            '<label>Name</label>\n\n<input type="text">\n\n<input type="submit">\n\n'
            "</form>\n```\n\nCommon input types:\n\n- text\n- email\n- password\n"
            "- checkbox\n- radio\n- number\n- date\n\n---\n\n## Semantic HTML\n\n"
            "Semantic HTML improves readability and accessibility.\n\nExamples:\n\n"
            "- header\n- nav\n- main\n- article\n- section\n- aside\n- footer\n\n"
            "Semantic elements help search engines understand webpage structure.\n\n"
            "---\n\n## Best Practices\n\n1. Use semantic HTML whenever possible.\n"
            "2. Keep code properly indented.\n3. Write meaningful alt text for images.\n"
            "4. Avoid unnecessary nested elements.\n5. Use headings in the correct order.\n"
            "6. Validate HTML before deployment.\n\n---\n\n## Summary"
        ),
        metadata={"filename": "html1.md"},
    )

    question = "why should I use meaningful tags instead of div everywhere"

    print(f"original chunk: {len(messy_chunk.page_content)} chars, 4 mixed sections")
    print(f"cross-encoder score on the whole chunk: -8.889 (threshold -3.5, rejected)\n")

    t0 = time.perf_counter()
    result = extract_relevant(question, messy_chunk)
    elapsed = time.perf_counter() - t0

    if result is None:
        print(f"extraction found nothing relevant ({elapsed:.2f}s) - compression doesn't fix this")
    else:
        print(f"extracted ({elapsed:.2f}s, {len(result.page_content)} chars):")
        print(f"  {result.page_content!r}")
