import os

# Must be set before langchain_community is imported - it reads USER_AGENT
# at import time and warns if it's missing. Some sites also reject requests
# that don't identify themselves.
os.environ.setdefault("USER_AGENT", "AI-Knowledge-Assistant/1.0")

from pathlib import Path  # noqa: E402

from langchain_community.document_loaders import (  # noqa: E402
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}


def is_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://"))


def get_loader(source: str):
    if is_url(source):
        return WebBaseLoader(source)

    extension = Path(source).suffix.lower()

    loaders = {
        ".txt": lambda: TextLoader(source, encoding="utf-8"),
        ".md": lambda: TextLoader(source, encoding="utf-8"),
        ".pdf": lambda: PyPDFLoader(source),
        ".docx": lambda: Docx2txtLoader(source),
        ".csv": lambda: CSVLoader(source),
    }

    if extension not in loaders:
        raise ValueError(f"unsupported file type:{extension}")

    return loaders[extension]()


def load_document(source: str):
    """Load a local file or a URL into a list of Documents.

    A web page arrives as one Document holding the whole page, including
    whatever navigation and footer text the site has - WebBaseLoader
    strips tags, not boilerplate. Blank lines are collapsed here so that
    doesn't turn into a chunk of empty space.
    """
    docs = get_loader(source).load()

    if is_url(source):
        for doc in docs:
            doc.page_content = "\n".join(
                line.strip() for line in doc.page_content.splitlines() if line.strip()
            )

    return docs


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else input("file path or URL: ").strip()

    if not is_url(source) and not Path(source).exists():
        print(f"Error: file '{source}' does not exist")
        raise SystemExit(1)

    try:
        docs = load_document(source)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raise SystemExit(1)

    print(f"\nLoaded {len(docs)} document(s) from {'URL' if is_url(source) else 'file'}\n")

    for i, doc in enumerate(docs, start=1):
        print(f"--- Document {i} ---")
        print("Metadata:", doc.metadata)
        preview = doc.page_content.strip()
        print(preview[:800] + ("\n...(truncated)" if len(preview) > 800 else ""))
        print()
