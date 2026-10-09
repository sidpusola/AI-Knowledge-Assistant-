"""Decide what kind of diagram a question needs, before answering it.

Asking the tutor model to both choose a diagram format and produce it
inside one long prompt measured 5/9 on a nine-question set: it reverts
to Mermaid for everything, because Mermaid is what it has seen most of.
Choosing is a classification problem, so it gets its own call with a
constrained output - a Literal becomes a JSON-schema enum, and the
decoder then cannot emit anything else.

Costs one extra call, on the fast model, generating a handful of
tokens - and only for questions that actually ask for a picture.
"""

import re
from functools import lru_cache
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm.provider import get_llm
from config import settings

DiagramKind = Literal[
    "cycle", "timeline", "linear", "decision_tree", "hierarchy", "parallel", "none"
]

# Which renderer each kind ends up in. cycle/timeline/linear have their
# own React renderers; the rest are graphs and go to Mermaid.
JSON_KINDS = {"cycle", "timeline", "linear"}


class DiagramPlan(BaseModel):
    """The chosen shape for one answer's diagram."""

    kind: DiagramKind = Field(
        description=(
            "cycle: repeats, last stage leads back to the first. "
            "timeline: events in time order, does not repeat. "
            "linear: one step causes the next and it ends. "
            "decision_tree: conditional logic, yes/no routes. "
            "hierarchy: parent/child classification. "
            "parallel: work that splits and rejoins. "
            "none: no diagram would help."
        )
    )
    theme: Literal["minimal", "biology", "technical", "education"] = Field(
        default="minimal",
        description=(
            "biology for living things, technical for computing and "
            "engineering, education for study and learning topics, "
            "minimal otherwise."
        ),
    )


_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an educational diagram planner. Given a student's "
        "question, choose the diagram shape that matches the CONTENT's "
        "structure, not the words they used.\n\n"
        "Judge by meaning and ignore the word the student used - "
        "'flowchart' says nothing about the shape.\n\n"
        "Worked examples:\n"
        "  menstrual cycle                -> cycle\n"
        "  life cycle of a butterfly      -> cycle\n"
        "  the water cycle                -> cycle\n"
        "  the nitrogen cycle             -> cycle\n"
        "  history of the internet        -> timeline\n"
        "  phases of a software project   -> timeline\n"
        "  steps to install Python        -> linear\n"
        "  how binary search works        -> decision_tree\n"
        "  troubleshooting a dead laptop  -> decision_tree\n"
        "  types of machine learning      -> hierarchy\n"
        "  parts of a plant cell          -> hierarchy\n"
        "  a RAG pipeline with routing    -> parallel\n"
        "  CI/CD build and test stages    -> parallel\n"
        "  what is HTML                   -> none\n\n"
        "Anything that returns to where it began is 'cycle', however it "
        "is phrased - that includes every named cycle in biology, "
        "chemistry and earth science. Answer 'none' only when a picture "
        "genuinely would not help, such as a plain definition.",
    ),
    ("human", "{question}"),
])

# Questions that are asking for a picture. The planner only runs for
# these, so an ordinary question pays nothing.
_DIAGRAM_WORDS = re.compile(
    r"\b(diagram|flow ?chart|flow|chart|draw|sketch|illustrate|visuali[sz]e|"
    r"timeline|life ?cycle|cycle|steps|stages|phases|process|pipeline|"
    r"architecture|workflow)\b",
    re.I,
)


def wants_diagram(question: str) -> bool:
    """Cheap pre-filter, deliberately generous - a false positive costs
    one short call on the fast model, a false negative costs a picture."""
    return bool(_DIAGRAM_WORDS.search(question or ""))


@lru_cache(maxsize=1)
def _planner():
    # The fast model: this is a single-token-ish choice, not reasoning,
    # and the structured-output schema does the constraining.
    return _PLANNER_PROMPT | get_llm(settings.OLLAMA_MODEL_FAST).with_structured_output(
        DiagramPlan
    )


def plan_diagram(question: str) -> DiagramPlan | None:
    """The diagram shape for this question, or None to leave it alone."""
    if not wants_diagram(question):
        return None

    try:
        plan = _planner().invoke({"question": question})
    except Exception:
        # A planner failure must never cost the student their answer -
        # the tutor prompt still describes both formats and will pick
        # something reasonable on its own.
        return None

    if plan is None or plan.kind == "none":
        return None
    return plan


def plan_directive(plan: DiagramPlan | None) -> str:
    """The line injected into the tutor prompt. Empty when there is no
    plan, which leaves the prompt exactly as it was."""
    if plan is None:
        return ""

    if plan.kind in JSON_KINDS:
        return (
            "DIAGRAM DECISION (already made for you, follow it exactly): "
            f'this answer needs a ```flow-json block with "diagram_type": '
            f'"{plan.kind}" and "theme": "{plan.theme}". Do not write a '
            "```mermaid block."
        )
    return (
        "DIAGRAM DECISION (already made for you, follow it exactly): this "
        f"answer needs a ```mermaid block laid out as a {plan.kind.replace('_', ' ')}. "
        "Do not write a ```flow-json block."
    )


if __name__ == "__main__":
    for q in [
        "flowchart of menstrual cycle",
        "give me a timeline of the history of AI",
        "show the steps to install Python as a diagram",
        "diagram the types of machine learning",
        "flowchart for troubleshooting a laptop that won't turn on",
        "what is HTML?",
    ]:
        plan = plan_diagram(q)
        print(f"  {str(plan.kind if plan else None):14} "
              f"{str(plan.theme if plan else '-'):10} {q}")
