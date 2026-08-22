from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter():
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    return splitter

def split_documents(documents):
    splitter=get_text_splitter()
    chunks=splitter.split_documents(documents)

    return chunks



if __name__=="__main__":
    from rag.loaders import load_document
    docs=load_document(r"C:\Users\sidpu\OneDrive\Desktop\Ai knowledge assistant\docs\html1.md")
    chunks=split_documents(docs)
    print(f"Original documents:{len(docs)}")
    print(f"Chunks:{len(chunks)}")

    for i,chunk in enumerate(chunks):
        print("\n"+"="*60)
        print(f"chunk {i+1}")
        print("="*60)
        print(chunk.page_content)