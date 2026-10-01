
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

    topic: str = Field(description="The specific topic being taught this turn")
    explanation: str = Field(description="Simple explanation of the topic")
    follow_up_question: str = Field(
        description="One question to check the student's understanding"
    )


class BasicChat:
    """A minimal multi-turn chatbot with no retrieval"""

    def __init__(self):
        self.history: list[HumanMessage | AIMessage] = []

    def ask(self, question: str) -> TutorResponse:
        llm = route_llm(question)

        structured_llm = llm.with_structured_output(TutorResponse)
        chain = prompt | structured_llm

        response = cast(TutorResponse, safe_invoke(chain, {
            "question": question,
            "history": self.history,
        }))

        self.history.append(HumanMessage(content=question))
        
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
