from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(
        model="qwen2.5:7b",
        temperature=0.3,
    )
    

if __name__=="__main__":
    llm=get_llm()
    response=llm.invoke(
        "Explain HTML is one simple sentence "
    )
    print(response.content)

    