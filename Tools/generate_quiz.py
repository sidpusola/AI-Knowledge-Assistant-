from llm.provider import get_llm

llm=get_llm()
def generate_quiz(topic):
    return llm.invoke(topic)
                  