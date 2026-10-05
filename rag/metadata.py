

import hashlib
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.documents import Document

# Loader-supplied keys that describe the authoring tool rather than the
# content. Dropped so they can't crowd out or be confused with ours.
_NOISE_KEYS = {
    "producer",
    "creator",
    "creationdate",
    "moddate",
    "author",
    "title",
    "subject",
    "keywords",
}

# Loader keys genuinely worth keeping - page numbers make a citation
# possible ("page 3 of the lecture") rather than just naming the file.
_KEEP_KEYS = {"page", "page_label", "total_pages", "source", "row"}

# "title" is in _NOISE_KEYS because a PDF written in Word carries whatever
# was left in its document properties - typically an untouched placeholder.
# A web page's <title> is the opposite: it's the single most useful label
# the page has. Same key, opposite value, so it's kept only for web sources.
_WEB_KEEP_KEYS = _KEEP_KEYS | {"title", "language"}


def document_id(path: str) -> str:
    """Stable id for a document, derived from its filename.

    Deliberately the filename and not the content: re-uploading an edited
    version of the same file should be recognisable as the same document,
    which is what makes replacing rather than duplicating it possible.

    For a URL the whole URL is hashed, not just its last segment -
    two sites can both end in "/guide" and they are not the same document.
    """
    from rag.loaders import is_url

    key = path if is_url(path) else Path(path).name
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_document_metadata(path: str, **extra: Any) -> dict[str, Any]:
    """Metadata describing the document as a whole.

    Anything passed as **extra (category, topic, difficulty, department)
    is merged in, with None values dropped for Chroma's sake.
    """
    from rag.loaders import is_url

    if is_url(path):
        # A URL is not a filesystem path. Path(url).suffix would give
        # ".com" for a bare domain and "" for an extensionless page, so
        # the host plus path is used as the display name instead.
        parsed = urlparse(path)
        name = f"{parsed.netloc}{parsed.path}".rstrip("/") or parsed.netloc
        file_type = "web"
    else:
        source = Path(path)
        name = source.name
        file_type = source.suffix.lower().lstrip(".")

    metadata: dict[str, Any] = {
        "doc_id": document_id(path),
        "filename": name,
        "file_type": file_type,
        "ingested_at": date.today().isoformat(),
    }

    for key, value in extra.items():
        if value is not None:
            metadata[key] = value

    return metadata


def enrich_chunks(chunks: list[Document], path: str, **extra: Any) -> list[Document]:
    """Attach clean, useful metadata to every chunk from one document.

    Chunk-level position (index and total) is added alongside the
    document-level fields, so a retrieved chunk can say where it came
    from and roughly how far into the document it sits.
    """
    from rag.loaders import is_url

    document_metadata = build_document_metadata(path, **extra)
    total = len(chunks)
    keep_keys = _WEB_KEEP_KEYS if is_url(path) else _KEEP_KEYS

    for index, chunk in enumerate(chunks):
        kept = {
            key: value
            for key, value in chunk.metadata.items()
            if key in keep_keys and value is not None
        }

        chunk.metadata = {
            **kept,
            **document_metadata,
            "chunk_index": index,
            "chunk_count": total,
        }

    return chunks


def describe(chunk: Document) -> str:
    """One-line human description of where a chunk came from.

    Intended for source citations in an answer - the point of carrying
    this metadata at all.
    """
    name = chunk.metadata.get("filename", "unknown")
    page = chunk.metadata.get("page")

    if page is not None:
        return f"{name}, page {int(page) + 1}"
    return name


def build_citations(chunks: list[Document]) -> list[str]:
    """Unique, ordered source descriptions for a list of retrieved chunks.

    Several chunks commonly come from the same page, so mapping describe()
    straight over the list would repeat "notes.pdf, page 3" three times -
    this keeps first-seen order and drops the repeats.
    """
    seen: list[str] = []
    for chunk in chunks:
        label = describe(chunk)
        if label not in seen:
            seen.append(label)
    return seen


if __name__ == "__main__":
    from rag.loaders import load_document
    from rag.splitter import split_documents

    path = "uploads/cognizent tech.pdf"
    chunks = split_documents(load_document(path))

    print("BEFORE (whatever the loader emitted):")
    print(" ", chunks[0].metadata)
    print()

    enrich_chunks(chunks, path, category="career", topic="placement assessment")

    print("AFTER:")
    print(" ", chunks[0].metadata)
    print()
    print("citation:", describe(chunks[0]))
