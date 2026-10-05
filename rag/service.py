"""
LangChain + ChromaDB Vector Search Module - the Week 4 deliverable.

The capabilities already existed, spread across three modules: ingestion
in rag/ingest.py, storage and deletion in rag/vectorstore.py, search and
filtering in rag/retriever.py. Each is sensible on its own, but together
they aren't a thing you can hand to someone - adding a document and then
searching for it meant importing from three places and knowing which
function belonged where.

This is one object with the whole lifecycle on it:

    add      a file or URL, through the full pipeline
    search   with optional subject / type filtering and relevance scores
    list     what is actually indexed, per document rather than per chunk
    delete   every chunk of a document
    stats    what the collection currently holds

It delegates rather than reimplements - the logic stays where it was,
this gives it one front door.

    service = VectorStoreService()
    service.add("uploads/notes.pdf")
    service.search("what is a data warehouse", category="computer science")
    service.delete(doc_id)

The vectorstore is loaded once and reused. That matters: load_vectorstore()
rebuilds the embedding model each call, which measured at ~8-10s.
"""

from typing import Any

from langchain_core.documents import Document

from config import settings
from rag.ingest import ingest_document
from rag.metadata import describe, document_id
from rag.retriever import get_retriever, search_with_scores
from rag.vectorstore import delete_document, list_indexed_documents, load_vectorstore


class VectorStoreService:
    """One interface over the Chroma collection and its pipeline."""

    def __init__(self):
        self._vectorstore = None

    @property
    def vectorstore(self):
        """Loaded on first use and kept, since building it is expensive."""
        if self._vectorstore is None:
            self._vectorstore = load_vectorstore()
        return self._vectorstore

    # --- adding -------------------------------------------------------

    def add(self, source: str) -> dict[str, Any]:
        """Run a file or URL through load -> split -> metadata -> store.

        Re-adding something already indexed replaces it rather than
        duplicating it, keyed on doc_id.
        """
        chunks = ingest_document(source)

        return {
            "source": source,
            "doc_id": document_id(source),
            "chunks": chunks,
            "indexed": chunks > 0,
        }

    # --- searching ----------------------------------------------------

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        file_type: str | None = None,
        k: int | None = None,
        with_scores: bool = False,
    ) -> list:
        """Similarity search, optionally restricted by metadata.

        with_scores returns (Document, relevance) pairs and bypasses the
        relevance floor, so you can see what was rejected and why - the
        plain path applies the floor and returns documents only.
        """
        if with_scores:
            return search_with_scores(
                self.vectorstore, query, category=category, file_type=file_type, k=k
            )

        retriever = get_retriever(
            self.vectorstore, category=category, file_type=file_type, k=k
        )
        return retriever.invoke(query)

    # --- managing -----------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """One entry per indexed document, with its chunk count."""
        return list_indexed_documents()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        for document in self.list_documents():
            if document["doc_id"] == doc_id:
                return document
        return None

    def delete(self, doc_id: str) -> int:
        """Remove a document. Returns how many chunks were deleted."""
        return delete_document(doc_id)

    # --- inspecting ---------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """What the collection currently holds."""
        documents = self.list_documents()
        chunks = sum(d["chunks"] for d in documents)

        by_category: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for document in documents:
            by_category[document["category"]] = by_category.get(document["category"], 0) + 1
            by_type[document["file_type"]] = by_type.get(document["file_type"], 0) + 1

        return {
            "documents": len(documents),
            "chunks": chunks,
            "by_category": by_category,
            "by_file_type": by_type,
            "embedding_model": settings.EMBEDDING_MODEL,
            "persist_directory": settings.CHROMA_DB_PATH,
            "relevance_threshold": settings.RETRIEVER_SCORE_THRESHOLD,
        }

    @staticmethod
    def cite(document: Document) -> str:
        """Human-readable source for a retrieved chunk."""
        return describe(document)


# A module-level instance, so callers share one loaded vectorstore rather
# than each paying to build their own.
service = VectorStoreService()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m rag.service")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("stats")
    sub.add_parser("list")

    p_add = sub.add_parser("add")
    p_add.add_argument("source", help="file path or URL")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--category")
    p_search.add_argument("--file-type")
    p_search.add_argument("--scores", action="store_true")

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("doc_id")

    args = parser.parse_args()

    if args.command == "stats":
        print(json.dumps(service.stats(), indent=2))

    elif args.command == "list":
        for d in service.list_documents():
            print(
                f"{d['doc_id']}  {d['filename'][:40]:42}{d['file_type']:5}"
                f"{d['category']:18}{d['chunks']:4} chunks"
            )

    elif args.command == "add":
        print(json.dumps(service.add(args.source), indent=2))

    elif args.command == "search":
        results = service.search(
            args.query,
            category=args.category,
            file_type=args.file_type,
            with_scores=args.scores,
        )
        if not results:
            print("nothing cleared the relevance floor")
        for item in results:
            if args.scores:
                document, score = item
                print(f"  {score:6.3f}  {service.cite(document)}")
            else:
                print(f"  {service.cite(item)}")

    elif args.command == "delete":
        print(f"deleted {service.delete(args.doc_id)} chunks")

    else:
        parser.print_help()
