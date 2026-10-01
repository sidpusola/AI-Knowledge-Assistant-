"""
Classification chain - Week 2 deliverable #2.

Demonstrates *constrained* structured output, which is the part that
differs from summarize.py. There, the schema asks for a string and the
model fills it with anything it likes. Here the schema restricts the
allowed values with typing.Literal, which becomes an enum in the JSON
schema handed to Ollama - so the model physically cannot return
"Computer Science!" or "cs/career" or a polite paragraph. Exactly one of
the listed values comes back, every time.

That's what makes the result safe to branch on in code instead of
string-matching hopefully.

Beyond the exercise, this is the missing piece for the "filter knowledge
by department" requirement in the final product spec: run it at ingest
time, write the category into Chroma metadata, and department filtering
becomes a metadata filter on retrieval.
"""

from typing import Literal, cast

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm.provider import get_llm, safe_invoke

# Kept as a module-level alias so callers (and later, the ingest pipeline)
# can reference the same set rather than duplicating the string literals.
Subject = Literal[
    "computer science",
    "mathematics",
    "life skills",
    "career",
    "other",
]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You classify study material for a student knowledge base.

Read the text and decide which single subject it belongs to:

- computer science: programming, web development, databases, networks, AI
- mathematics: algebra, calculus, statistics, discrete maths, proofs
- life skills: ethics, values, relationships, communication, wellbeing
- career: placements, interviews, resumes, company assessments, job roles
- other: anything that genuinely fits none of the above

Pick the single best fit. Say how confident you are, and give one short
sentence explaining the choice.
""",
    ),
    ("human", "{text}"),
])


class ClassificationResult(BaseModel):
    # Constrained - this is the one code branches on and filters by.
    category: Subject = Field(description="The single best-fit subject for this text")

    # Free text - five buckets are far too coarse to describe real study
    # material, so the specific topic is captured here instead of being
    # forced into the taxonomy. Filter on `category`, show `topic`.
    # It also makes a bad taxonomy fit visible: topic "Italian cooking"
    # under category "life skills" is obviously a stretch, where the
    # category alone would have hidden it.
    topic: str = Field(description="The specific topic, in a few words")

    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Roughly how advanced this material is"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How clearly the text fits the chosen category"
    )
    reasoning: str = Field(
        description="One short sentence explaining why this category was chosen"
    )


def classify_text(text: str) -> ClassificationResult:
    llm = get_llm().with_structured_output(ClassificationResult)
    chain = prompt | llm
    return cast(ClassificationResult, safe_invoke(chain, {"text": text}))


if __name__ == "__main__":
    from pathlib import Path

    from rag.loaders import load_document

    # Classify whatever is actually in uploads/, so the demo reflects real
    # material rather than hand-picked sample strings.
    upload_dir = Path("uploads")
    documents = sorted(p for p in upload_dir.iterdir() if p.is_file()) if upload_dir.exists() else []

    if not documents:
        print("No documents in uploads/ - upload something first.")

    for path in documents:
        try:
            pages = load_document(str(path))
        except Exception as e:
            print(f"{path.name}: could not read ({e})")
            continue

        # The opening pages are enough to tell the subject, and keep the
        # prompt small for a big PDF.
        text = "\n\n".join(page.page_content for page in pages)[:3000]

        result = classify_text(text)
        print(f"{path.name}")
        print(f"  -> {result.category} / {result.topic} ({result.difficulty})")
        print(f"     {result.reasoning}")
        print()
