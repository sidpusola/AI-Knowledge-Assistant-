"""
ChromaDB access - adding, replacing and deleting documents.

The collection is persistent (CHROMA_DB_PATH), so it survives restarts.

A note on re-ingesting. Chunk ids are derived from the source, the chunk's
position and its content, so editing a document changes the ids of every
chunk from the edit onward. Adding those without removing the old ones
leaves both versions in the store - the stale text doesn't just waste
space, it competes with the current text at retrieval time, so a question
about an updated policy can be answered from the version it replaced.

ingest_vectorstore therefore deletes a document's existing chunks before
adding the new ones, keyed on the doc_id that rag/metadata.py stamps.
"""

import hashlib

from langchain_chroma import Chroma

from config import settings

from .embeddings import get_embeddings

DB_path = settings.CHROMA_DB_PATH


def make_chunk_id(source, index, content):
    raw = f"{source}:{index}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_vectorstore():
    return Chroma(
        persist_directory=DB_path,
        embedding_function=get_embeddings(),
    )


def delete_document(doc_id: str) -> int:
    """Remove every chunk belonging to one document.

    Returns how many chunks were deleted - 0 if that doc_id wasn't in the
    store, which lets the caller tell "nothing was there" apart from
    "something was replaced".
    """
    vectorstore = load_vectorstore()

    existing = vectorstore.get(where={"doc_id": doc_id})
    ids = existing.get("ids", [])

    if ids:
        vectorstore.delete(ids=ids)

    return len(ids)


def ingest_vectorstore(chunks, replace: bool = True):
    """Store chunks, replacing any previous version of the same document.

    replace=False appends without clearing, which is only correct when
    you know the document isn't already indexed.
    """
    if not chunks:
        return load_vectorstore()

    vectorstore = load_vectorstore()

    doc_id = chunks[0].metadata.get("doc_id")
    if replace and doc_id:
        removed = delete_document(doc_id)
        if removed:
            print(f"  (replaced {removed} existing chunks)")

    ids = [
        make_chunk_id(chunk.metadata.get("source", "unkown"), index, chunk.page_content)
        for index, chunk in enumerate(chunks)
    ]

    vectorstore.add_documents(documents=chunks, ids=ids)
    return vectorstore


def list_indexed_documents() -> list[dict]:
    """One entry per document actually in the store, not per chunk.

    This is what the app should report as its knowledge base - listing
    files on disk instead says nothing about what is searchable, and
    misses anything ingested from a URL.
    """
    records = load_vectorstore().get()
    documents: dict[str, dict] = {}

    for metadata in records.get("metadatas", []):
        doc_id = metadata.get("doc_id")
        if not doc_id:
            continue

        entry = documents.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "filename": metadata.get("filename", "unknown"),
                "file_type": metadata.get("file_type", ""),
                "category": metadata.get("category", "uncategorised"),
                "source": metadata.get("source", ""),
                "ingested_at": metadata.get("ingested_at", ""),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1

    return sorted(documents.values(), key=lambda d: d["filename"])


if __name__ == "__main__":
    for document in list_indexed_documents():
        print(
            f"{document['filename'][:45]:47}{document['file_type']:6}"
            f"{document['category']:18}{document['chunks']:4} chunks"
        )
