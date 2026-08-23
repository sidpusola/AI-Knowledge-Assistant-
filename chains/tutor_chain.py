from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from rag.vectorstore import load_vectorstore
from rag.retriever import get_retriever
from llm.model import get_llm

prompt=ChatPromptTemplate.from_messages([
    (
    "system",
    """
You are an AI tutor.

Answer the student's question using only the provided context.

Explain the answer clearly and simply.

If the context does not contain enough information,
say that you do not have enough information.

context:
{context}
"""
    ),
    (
       "human",
       "{question}"
    )
])

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


def get_tutor_chain():
    vectorstore=load_vectorstore()

    retriever=get_retriever(vectorstore)

    llm=get_llm()

    chain=(
         {
              "context":retriever|format_docs,
              "question":RunnablePassthrough()

         }|prompt
          |llm
          |StrOutputParser()
    )
    return chain

def ask_tutor(question:str):
     chain=get_tutor_chain()
     answer=chain.invoke(question)
     return answer


if __name__=="__main__":
      answer=ask_tutor(
         "what is semantic HTML?"
      )
      print(answer)