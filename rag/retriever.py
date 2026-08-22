def get_retriever(vectorstore):

    retriever=vectorstore.as_retriever(
        search_kwargs={
                     "k":3
               }
    )
    return retriever

if __name__=="__main__":
    from rag.vectorstore import load_vectorstore

    vectorstore=load_vectorstore()
    retriever=get_retriever(vectorstore)
    results=retriever.invoke("what is semantic html")

    for i,doc in enumerate(results,start=1):
        print(f"\n--- Result{i} ---")
        print(doc.page_content)