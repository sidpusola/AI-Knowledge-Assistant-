"""
Text splitting, with a choice of how chunk size is measured.

Characters are the easy unit but the wrong one. The embedding model
(all-MiniLM-L6-v2) accepts 256 *tokens* and silently truncates anything
longer - no error, the tail of the chunk just never makes it into the
vector. So the quantity that matters is tokens, and characters are only
a proxy for it.

Measured across the documents currently in this project, 500-character
chunks produce between 3 and 202 tokens. Same setting, 60x spread,
because the character-to-token ratio depends entirely on the text. That
is the problem with the character-based splitter: it cannot tell you how
close you are to the limit, so you either leave most of the budget
unused (the average chunk here is ~85 tokens of 256) or risk silent
truncation on dense text.

Three options, set with SPLITTER_TYPE:

  recursive        RecursiveCharacterTextSplitter, sized in characters.
                   The original behaviour. Splits on paragraph, then
                   sentence, then word boundaries, so chunks read well.

  token            SentenceTransformersTokenTextSplitter, sized in
                   tokens using this embedding model's own tokenizer.
                   Exact token control, but it cuts at token boundaries
                   with no regard for sentences - a chunk can end
                   mid-sentence. (Not TokenTextSplitter, which measures
                   with OpenAI's tiktoken and so counts the wrong thing
                   for a sentence-transformers model.)

  recursive_token  RecursiveCharacterTextSplitter driven by the
                   embedding model's own tokenizer. Keeps the sentence-
                   aware splitting *and* measures in tokens. Usually what
                   you want; the only cost is loading the tokenizer.
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from config import settings


def _huggingface_tokenizer():
    """The tokenizer the embedding model itself uses.

    Deliberately the same one - measuring with a different tokenizer
    would give numbers that don't correspond to the limit being avoided.
    """
    from rag.embeddings import get_embeddings

    return get_embeddings()._client.tokenizer


def get_text_splitter():
    splitter_type = settings.SPLITTER_TYPE.lower()

    if splitter_type == "token":
        # Not TokenTextSplitter - that one measures with tiktoken, which is
        # OpenAI's tokenizer and not the one this embedding model uses.
        # Measuring against the wrong tokenizer defeats the point.
        return SentenceTransformersTokenTextSplitter(
            model_name=settings.EMBEDDING_MODEL,
            tokens_per_chunk=settings.CHUNK_SIZE_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
        )

    if splitter_type == "recursive_token":
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            _huggingface_tokenizer(),
            chunk_size=settings.CHUNK_SIZE_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )


def split_documents(documents):
    return get_text_splitter().split_documents(documents)


if __name__ == "__main__":
    # Compare the three strategies on a real document, in the unit that
    # actually matters.
    from rag.loaders import load_document

    LIMIT = 256
    tokenizer = _huggingface_tokenizer()
    docs = load_document("docs/html1.md")

    original = settings.SPLITTER_TYPE
    print(f"embedding model budget: {LIMIT} tokens\n")
    print(f"{'strategy':18}{'chunks':>8}{'min':>6}{'avg':>6}{'max':>6}{'over limit':>12}")

    for strategy in ("recursive", "token", "recursive_token"):
        settings.SPLITTER_TYPE = strategy
        chunks = split_documents(docs)
        counts = [len(tokenizer.encode(c.page_content)) for c in chunks]
        over = sum(1 for n in counts if n > LIMIT)
        print(
            f"{strategy:18}{len(chunks):>8}{min(counts):>6}"
            f"{sum(counts) // len(counts):>6}{max(counts):>6}{over:>12}"
        )

    settings.SPLITTER_TYPE = original
