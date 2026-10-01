"""
Email drafting chain - Week 2 deliverable #5.

Demonstrates prompt chaining: two LLM calls in sequence, where the first
call's structured output becomes part of the second call's prompt. That's
the distinction from every other chain here - `prompt | llm | parser` is
one model call however long the pipeline looks, while this is genuinely
two, and the second cannot start until the first has answered.

    request
       |
       v
    analyse  -> EmailIntent(purpose, tone, key_points, recipient)   [call 1]
       |
       v
    draft    -> EmailDraft(subject, body)                           [call 2]

Why bother, rather than asking for the email in one shot: the first call
decides *what the email needs to do* before any wording exists, so the
second call writes against a settled brief instead of inventing the goal
and the prose at the same time. It also makes the intermediate decision
visible - you can see the tone it picked and why the email came out the
way it did, which a single call hides.

The cost is real though: two calls means roughly twice the latency. For
a short email that is a poor trade, and one well-written prompt would do.
It earns its keep when the brief genuinely needs deciding - which is the
honest way to describe when prompt chaining is worth reaching for.

RunnablePassthrough.assign() is what threads it together: it adds a new
key to the dict flowing through the chain while keeping what was already
there, so the drafting step can see both the original request and the
analysis of it.
"""

from typing import Any, Literal, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from llm.provider import get_llm, safe_invoke


class EmailIntent(BaseModel):
    purpose: str = Field(description="What this email needs to achieve, in one sentence")
    recipient: str = Field(description="Who it is addressed to, e.g. 'professor', 'recruiter'")
    tone: Literal["formal", "friendly", "urgent"] = Field(
        description="The tone appropriate for this recipient and purpose"
    )
    key_points: list[str] = Field(
        description="2-4 points the email must cover for it to do its job"
    )


class EmailDraft(BaseModel):
    subject: str = Field(description="Subject line - specific, not generic")
    body: str = Field(description="The email body, including greeting and sign-off")


analyse_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
A student wants to send an email. Work out what the email actually needs
to do before anything is written.

Decide its purpose, who it is going to, the tone that fits, and the
points it must cover. Do not write the email itself - only the brief.

Give at most 4 key points, and only ones the student's request actually
supports. Never add a point that asks for detail the student did not
provide - "state the specific reason for your illness" is exactly the
kind of instruction that makes the next step invent facts.
""",
    ),
    ("human", "{request}"),
])


draft_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Write the email described by the brief below. Follow the brief - it has
already been decided.

Purpose: {purpose}
Recipient: {recipient}
Tone: {tone}
Must cover:
{key_points}

Keep it short - a student's email, not a press release.

Use only facts the student actually gave you. Do not invent specifics to
fill gaps - no made-up illnesses, dates, grades or reasons. Where a
detail is needed but unknown, leave a placeholder in brackets such as
[Name], [date] or [reason] for the student to fill in. An email they have
to complete is fine; one that states something untrue on their behalf is
not.
""",
    ),
    ("human", "{request}"),
])


def _brief_to_prompt_vars(data: dict) -> dict:
    """Flatten the EmailIntent from call 1 into the variables call 2 needs."""
    intent: EmailIntent = data["intent"]
    return {
        "request": data["request"],
        "purpose": intent.purpose,
        "recipient": intent.recipient,
        "tone": intent.tone,
        "key_points": "\n".join(f"- {point}" for point in intent.key_points),
    }


def draft_email(request: str) -> dict[str, Any]:
    """Draft an email from a plain-language request.

    Returns {"request": ..., "intent": EmailIntent, "draft": EmailDraft} -
    the brief is returned alongside the email so the reasoning behind the
    wording is inspectable, not just the result.
    """
    llm = get_llm()

    analyse = analyse_prompt | llm.with_structured_output(EmailIntent)
    draft = (
        RunnableLambda(_brief_to_prompt_vars)
        | draft_prompt
        | llm.with_structured_output(EmailDraft)
    )

    # assign() adds a key to the dict without dropping the rest, so the
    # drafting step still has the original request alongside the brief.
    chain = (
        RunnablePassthrough.assign(intent=analyse)
        | RunnablePassthrough.assign(draft=draft)
    )

    return cast(dict[str, Any], safe_invoke(chain, {"request": request}))


if __name__ == "__main__":
    result = draft_email(
        "I need to email my professor asking for a one week extension on my "
        "database assignment because I was ill last week."
    )

    intent: EmailIntent = result["intent"]
    draft: EmailDraft = result["draft"]

    print("--- call 1: the brief ---")
    print(f"purpose:   {intent.purpose}")
    print(f"recipient: {intent.recipient}")
    print(f"tone:      {intent.tone}")
    print("must cover:")
    for point in intent.key_points:
        print(f"  - {point}")

    print()
    print("--- call 2: the email ---")
    print(f"Subject: {draft.subject}")
    print()
    print(draft.body)
