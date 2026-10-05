"""
The document ingestion pipeline.

    Document  ->  Loader  ->  Splitter  ->  Metadata  ->  Vector store
                              (rag/loaders, rag/splitter, rag/metadata)

The metadata step is what turns a chunk from "some text that matched"
into something a retrieved answer can cite and a query can filter on.
"""

from config import settings
from rag.loaders import load_document
from rag.metadata import enrich_chunks
from rag.splitter import split_documents
from rag.vectorstore import ingest_vectorstore


def _classify_document(text: str) -> dict:
    """Work out the subject of a document so it can be filtered on later.

    Imported lazily and wrapped: a classification failure should not cost
    you the upload. Losing the category means losing a filter; losing the
    ingest means losing the document.
    """
    if not settings.CLASSIFY_ON_INGEST:
        return {}

    try:
        from chains.classify import classify_text

        # The opening pages are enough to tell the subject, and keep the
        # prompt small for a large PDF.
        result = classify_text(text[:3000])
        return {
            "category": result.category,
            "topic": result.topic,
            "difficulty": result.difficulty,
        }
    except Exception as e:
        print(f"  (could not classify: {type(e).__name__} - ingesting without a category)")
        return {}


def ingest_document(path: str) -> int:
    """Load, split, enrich and store one document. Returns the chunk count."""
    docs = load_document(path)
    chunks = split_documents(docs)

    if not chunks:
        print("Indexed 0 chunks (no readable text found)")
        return 0

    full_text = "\n\n".join(doc.page_content for doc in docs)
    enrich_chunks(chunks, path, **_classify_document(full_text))

    ingest_vectorstore(chunks)

    category = chunks[0].metadata.get("category", "uncategorised")
    print(f"Indexed {len(chunks)} chunks from {chunks[0].metadata['filename']} [{category}]")
    return len(chunks)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "docs/html1.md"
    ingest_document(target)
