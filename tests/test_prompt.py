from llm.provider import get_llm
from llm.prompts import SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate
llm=get_llm()
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",SYSTEM_PROMPT),
        ("human","{question}")
    ]
)

chain=prompt|llm
response=chain.invoke(
    {
        "question":"Teach me HTML headings"
    }
)

print(response)