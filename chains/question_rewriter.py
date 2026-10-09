"""Turn a follow-up into a question that stands on its own.

Generation has had conversation memory since Week 5, but retrieval has
not: it embeds whatever the student typed. "What about the second one?"
is a perfectly clear thing to say and a useless thing to search for - it
shares no wording with any chunk, the relevance gate correctly rejects
everything, and the follow-up silently loses its grounding even though
the answer still reads fluently.

This rewrites such a question using the conversation so far, so the
*retriever* gets "what is a snowflake schema in a data warehouse?"
while the *generator* still receives what the student actually asked.

Most questions need nothing doing to them, so a cheap test runs first
and the LLM is only called for the ones that genuinely depend on what
came before.
"""

import re
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from llm.provider import get_llm, is_small_talk

# How much of the conversation the rewriter sees. The referent of "it" is
# almost always in the last exchange or two, and a longer window costs
# prefill on every follow-up.
HISTORY_WINDOW = 4

# Words that point at something said earlier. A question containing one
# of these cannot be searched for as written.
_REFERRING = {
    "it", "its", "it's", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    "he", "him", "his", "she", "her", "hers",
    "one", "ones", "former", "latter", "above", "previous", "earlier",
    "same", "such", "both", "either", "another", "others",
}

# Ordinals point backwards in "the second?" and forwards in "the first
# law of thermodynamics". Only the short form is a follow-up, so these
# count only in a brief question.
_ORDINALS = {"first", "second", "third", "fourth", "fifth", "last", "next"}
_ORDINAL_MAX_WORDS = 6

# Openers that are grammatically incomplete on their own.
# "why is" and "why does" were here and were wrong: they open plenty of
# self-contained questions ("why is the sky blue?"). The ones that
# genuinely cannot stand alone are left.
_ELLIPTICAL = (
    "what about", "how about", "and what", "and how", "and why",
    "what else", "anything else", "tell me more", "more on", "more about",
    "explain more", "go on", "continue", "why not",
    "and ", "but ", "then ",
)


def needs_rewrite(question: str, history: list) -> bool:
    """Whether this question depends on what came before.

    Deliberately generous: a false positive costs one short call on the
    fast model, a false negative costs the student their grounding. But
    it must not fire on a self-contained question, which is the common
    case and would otherwise pay that cost on every single turn.
    """
    if not history or not question:
        return False
    if is_small_talk(question):
        return False

    text = question.lower().strip()
    cleaned = re.sub(r"[^\w\s]", " ", text)
    words = cleaned.split()

    if any(word in _REFERRING for word in words):
        return True
    if text.startswith(_ELLIPTICAL):
        return True
    if len(words) <= _ORDINAL_MAX_WORDS and any(w in _ORDINALS for w in words):
        return True
    # A two-word question mid-conversation is a follow-up ("why?",
    # "any examples?"). The bar was four, which also caught plenty of
    # self-contained questions - "what is normalization?" is three -
    # and spent a model call proving they needed no rewriting.
    return len(words) <= 2


class RewrittenQuestion(BaseModel):
    """The follow-up, made self-contained."""

    question: str = Field(
        description=(
            "The student's question rewritten so it can be understood "
            "with no conversation around it. Keep their wording wherever "
            "it already works; only replace what refers backwards."
        )
    )


_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite a student's follow-up question so it can be "
        "understood on its own, for use as a search query over course "
        "materials.\n\n"
        "Replace anything that points backwards - it, they, that, the "
        "second one - with the thing it actually refers to, taken from "
        "the conversation. Keep the rest of their wording.\n\n"
        "Rules:\n"
        "- Return a question, not an answer, and never explain yourself.\n"
        "- Change as little as possible. If it already stands alone, "
        "return it unchanged.\n"
        "- Never invent a topic that is not in the conversation.\n\n"
        # Tried and reverted: an extra rule telling it to reuse the
        # conversation's exact nouns ("assessment", not "course"). It
        # left the score at 10/12 and made the phrasing clumsier -
        # "how does punishment use in operant conditioning?" - so the
        # 3b appears to spend its limited attention on whichever rule
        # came last. Not worth the regression.
        "Examples, given a conversation about data warehouses:\n"
        "  'what about the second one?'  -> 'what is a snowflake schema "
        "in a data warehouse?'\n"
        "  'why is it faster?'           -> 'why is a star schema "
        "faster?'\n"
        "  'what is normalization?'      -> 'what is normalization?'",
    ),
    ("placeholder", "{history}"),
    ("human", "Rewrite this follow-up as a standalone question: {question}"),
])


@lru_cache(maxsize=1)
def _rewriter():
    # Structured output so the result is the question itself, with no
    # "Sure, here's the rewritten version:" wrapped around it.
    return _REWRITE_PROMPT | get_llm(settings.OLLAMA_MODEL_FAST).with_structured_output(
        RewrittenQuestion
    )


def rewrite_question(question: str, history: list | None = None) -> str:
    """The question to retrieve with - rewritten if it needed it.

    Returns the original on every failure path. A bad rewrite is worse
    than no rewrite, and losing the student's answer to a rewriter
    hiccup would be far worse than either.
    """
    history = history or []
    if not needs_rewrite(question, history):
        return question

    try:
        result = _rewriter().invoke({
            "question": question,
            "history": history[-HISTORY_WINDOW:],
        })
    except Exception:
        return question

    if result is None:
        return question

    rewritten = (result.question or "").strip()
    if not rewritten:
        return question

    # Guard against the two ways this goes wrong in practice: the model
    # answering instead of rewriting, or padding a short question into a
    # paragraph. Either way the original is the safer query.
    if len(rewritten) > max(200, len(question) * 6):
        return question

    return rewritten


if __name__ == "__main__":
    history = [
        HumanMessage(content="what are the schema types in a data warehouse?"),
        AIMessage(content="The two main ones are the star schema and the "
                          "snowflake schema."),
    ]
    for q in [
        "what about the second one?",
        "why is it faster?",
        "what is normalization?",
        "thanks!",
        "the second?",
    ]:
        print(f"  {q!r:36} -> {rewrite_question(q, history)!r}")
