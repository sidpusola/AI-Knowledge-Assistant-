from rag.loaders import load_document
from rag.splitter import split_documents
from rag.vectorstore import ingest_vectorstore

def ingest_document(path):
    docs=load_document(path)
    chunks=split_documents(docs)
    vectorstore=ingest_vectorstore(chunks)
    print(f"Indexed {len(chunks)}chunks")
    return vectorstore

if __name__=="__main__":
    ingest_document(r"C:\Users\sidpu\OneDrive\Desktop\Ai knowledge assistant\docs\html1.md")