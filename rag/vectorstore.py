import hashlib
from langchain_chroma import Chroma
from .embeddings import get_embeddings
DB_path="./chroma_db"
def make_chunk_id(source,index,content):
    raw=f"{source}:{index}:{content}"   #combine them and then hash that string
    return hashlib.sha256(
        raw.encode()
        ).hexdigest()

def ingest_vectorstore(chunks):
    embeddings=get_embeddings()

    vectorstore=Chroma(
       embedding_function=embeddings,
        persist_directory=DB_path
    )
    ids=[]

    for i,chunk in enumerate(chunks):
        source=chunk.metadata.get("source","unkown")

        chunk_id=make_chunk_id(
            source,
            i,
            chunk.page_content
        )
        ids.append(chunk_id)

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )   
    return vectorstore

def load_vectorstore():
    embeddings=get_embeddings()
    vectorstore=Chroma(
        persist_directory=DB_path,
        embedding_function=embeddings
    )

    return vectorstore


if __name__=="__main__":
    from rag.loaders import load_document
    from rag.splitter import split_documents
    docs=load_document(r"C:\Users\sidpu\OneDrive\Desktop\Ai knowledge assistant\docs\html1.md")
    chunks=split_documents(docs)
    vector_store=ingest_vectorstore(chunks)
    print("database created successfully")