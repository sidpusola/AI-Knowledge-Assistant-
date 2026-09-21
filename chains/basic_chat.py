"""
Standalone chat chain - the Week 1 deliverable.

Deliberately has no retriever and no document context, unlike
chains/tutor_chain.py (that's the Week 5 RAG chain). This module exists
to demonstrate the Week 1 fundamentals in isolation:

- Chat models         -> llm.provider.route_llm
- Prompt templates     -> ChatPromptTemplate with a system + human message
- Message roles        -> system / human / ai, via MessagesPlaceholder
- Chains (LCEL)        -> prompt | llm.with_structured_output(...)
- Output parsers /
  structured response  -> TutorResponse (Pydantic), via with_structured_output

It also gives llm/prompts.py's SYSTEM_PROMPT its first real use - until
now it was only referenced by a manual test script, while tutor_chain.py
had its own separate, conflicting system prompt inline. Each chain now
has one clearly-owned persona: this one is the Socratic tutor
(SYSTEM_PROMPT), tutor_chain.py is the grounded document Q&A assistant.
"""

from typing import cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from llm.prompts import SYSTEM_PROMPT
from llm.provider import route_llm, safe_invoke

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


class TutorResponse(BaseModel):
    """Structured shape of a tutor turn - mirrors what SYSTEM_PROMPT
    already asks for in prose ("teach one topic, explain simply, then
    ask one question"), just made machine-readable instead of something
    a caller would have to re-parse out of free text.
    """

    topic: str = Field(description="The specific topic being taught this turn")
    explanation: str = Field(description="Simple explanation of the topic")
    follow_up_question: str = Field(
        description="One question to check the student's understanding"
    )


class BasicChat:
    """A minimal multi-turn chatbot with no retrieval.

    Keeps its own history as a list of HumanMessage/AIMessage objects -
    each call to ask() appends the new turn, so the next call sees the
    prior exchange via the "ai" and "human" message roles, not just a
    single stateless system+human prompt.
    """

    def __init__(self):
        self.history: list[HumanMessage | AIMessage] = []

    def ask(self, question: str) -> TutorResponse:
        # The model is picked per-question (see llm/provider.py:route_llm) -
        # short/simple questions get the fast model, longer/complex ones
        # get the general-purpose model, even mid-conversation.
        llm = route_llm(question)

        # with_structured_output makes the model itself return a parsed
        # TutorResponse instead of a raw string - no separate output
        # parser needed, and no risk of the model wrapping its answer in
        # commentary that then has to be stripped back out.
        structured_llm = llm.with_structured_output(TutorResponse)
        chain = prompt | structured_llm

        response = cast(TutorResponse, safe_invoke(chain, {
            "question": question,
            "history": self.history,
        }))

        self.history.append(HumanMessage(content=question))
        # Message history needs plain text, so the structured fields get
        # flattened back into one string for the next turn's context.
        self.history.append(AIMessage(
            content=f"{response.explanation}\n\n{response.follow_up_question}"
        ))

        return response

    def reset(self) -> None:
        self.history = []


if __name__ == "__main__":
    chat = BasicChat()

    turn1 = chat.ask("What is HTML?")
    print("[turn 1]")
    print("topic:", turn1.topic)
    print("explanation:", turn1.explanation)
    print("follow_up_question:", turn1.follow_up_question)
    print()

    turn2 = chat.ask(
        "Can you explain in more detail how a browser actually "
        "parses that HTML into the DOM, step by step?"
    )
    print("[turn 2]")
    print("topic:", turn2.topic)
    print("explanation:", turn2.explanation)
    print("follow_up_question:", turn2.follow_up_question)
