"""
Multi-Feature LangChain Utility Application - the Week 2 deliverable.

The five utility chains are each usable on their own, but on their own
they're five files with five import paths and five return types. This
turns them into one thing you can point at: a registry of named
utilities, a single run_utility() entry point, and a CLI.

    python -m chains.utility_hub --list
    python -m chains.utility_hub summarize --file docs/html1.md
    python -m chains.utility_hub classify --text "Binary search is O(log n)."

Two things here exist for what comes later rather than for the CLI:

- Each utility carries a `description`. That's the text an agent needs to
  choose between tools (Week 9), and what an intent router would match
  against. Written for a model to read, not just for --list.
- run_utility(name, text) is deliberately uniform. Every chain happens to
  take one string and return something serialisable, so a FastAPI route
  like /utilities/{name} is a thin wrapper over this rather than five
  hand-written endpoints (Week 8).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from chains.classify import classify_text
from chains.email_draft import draft_email
from chains.extract import extract_info
from chains.question_gen import generate_questions
from chains.summarize import summarize_text


@dataclass(frozen=True)
class Utility:
    name: str
    description: str
    input_label: str
    run: Callable[[str], Any]


UTILITIES: dict[str, Utility] = {
    "summarize": Utility(
        name="summarize",
        description=(
            "Condense a piece of study material into a short summary plus "
            "3-5 key points. Use when the student wants the gist of "
            "something long."
        ),
        input_label="the text to summarise",
        run=summarize_text,
    ),
    "classify": Utility(
        name="classify",
        description=(
            "Decide which subject a piece of material belongs to "
            "(computer science, mathematics, life skills, career, other), "
            "with its specific topic and difficulty. Use when tagging or "
            "filing material rather than answering a question about it."
        ),
        input_label="the text to classify",
        run=classify_text,
    ),
    "extract": Utility(
        name="extract",
        description=(
            "Pull structured facts out of material: defined terms, dates, "
            "named people and organisations, and key factual statements. "
            "Use when the student wants specifics rather than prose."
        ),
        input_label="the text to extract from",
        run=extract_info,
    ),
    "questions": Utility(
        name="questions",
        description=(
            "Generate practice questions from material - one factual, one "
            "conceptual, one applied - each with its expected answer. Use "
            "when the student wants to test themselves or revise."
        ),
        input_label="the material to generate questions from",
        run=generate_questions,
    ),
    "email": Utility(
        name="email",
        description=(
            "Draft an email from a plain-language description of what the "
            "student needs to send, deciding the purpose and tone first. "
            "Use for requests to professors, recruiters or administrators."
        ),
        input_label="a description of the email to write",
        run=draft_email,
    ),
}


def list_utilities() -> list[Utility]:
    return list(UTILITIES.values())


def run_utility(name: str, text: str) -> Any:
    """Run a named utility. Raises ValueError for an unknown name."""
    utility = UTILITIES.get(name)

    if utility is None:
        available = ", ".join(sorted(UTILITIES))
        raise ValueError(f"Unknown utility: {name!r}. Available: {available}")

    if not text.strip():
        raise ValueError(f"Nothing to work with - {name} needs {utility.input_label}.")

    return utility.run(text)


def serialise(value: Any) -> Any:
    """Make a utility's result JSON-friendly.

    The chains return different shapes - a Pydantic model, a dict holding
    Pydantic models, a model containing other models - so this walks
    whatever comes back rather than each caller special-casing per
    utility. It's also what a FastAPI route would need.
    """
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: serialise(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialise(item) for item in value]
    return value


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="python -m chains.utility_hub",
        description="Run one of the LangChain utility workflows.",
    )
    parser.add_argument("utility", nargs="?", choices=sorted(UTILITIES), help="which utility to run")
    parser.add_argument("--text", help="input text")
    parser.add_argument("--file", help="read input from a file instead of --text")
    parser.add_argument("--list", action="store_true", help="list the available utilities")

    args = parser.parse_args()

    if args.list or not args.utility:
        for utility in list_utilities():
            print(f"{utility.name:11} {utility.description}")
        sys.exit(0)

    if args.file:
        source = Path(args.file)
        if not source.exists():
            print(f"No such file: {source}")
            sys.exit(1)
        # Route through the document loaders so PDFs and DOCX work here
        # too, not just plain text files.
        from rag.loaders import load_document

        text = "\n\n".join(page.page_content for page in load_document(str(source)))
    elif args.text:
        text = args.text
    else:
        print("Give it something to work on: --text '...' or --file path")
        sys.exit(1)

    try:
        result = run_utility(args.utility, text)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(json.dumps(serialise(result), indent=2, ensure_ascii=False))
