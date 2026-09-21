import httpx
from langchain_ollama import ChatOllama
from ollama import ResponseError as OllamaResponseError

from config import settings


class LLMUnavailableError(Exception):
    """The configured LLM backend could not be reached or failed to respond.

    Covers the two realistic local-Ollama failure modes: the server isn't
    running (connection refused/timeout), or the requested model hasn't
    been pulled. Chains should invoke through safe_invoke() so callers
    (the FastAPI layer) only ever need to catch this one exception type,
    instead of knowing about httpx's or the ollama package's internals.
    """


def safe_invoke(chain, chain_input):
    """Invoke an LCEL chain, translating known Ollama/network failures
    into LLMUnavailableError.

    Verified against the real failure modes (not guessed): pointing
    ChatOllama at an unreachable port raises httpx.ConnectError, and
    invoking a model that isn't pulled raises ollama.ResponseError with a
    404. Both get normalized here into one clear, user-facing message.
    """
    try:
        return chain.invoke(chain_input)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise LLMUnavailableError(
            "Could not reach the local Ollama server. Is it running? "
            "(try: ollama serve)"
        ) from e
    except OllamaResponseError as e:
        raise LLMUnavailableError(
            f"Ollama returned an error: {e}. Is the model pulled? "
            "(try: ollama pull <model>)"
        ) from e


def _build_ollama(model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=settings.OLLAMA_TEMPERATURE,
    )


def get_llm(model: str | None = None):
    """Return a chat model for an explicitly-named model (or the default).

    This is the single place a chat model gets constructed - provider
    selection is driven by config.py's LLM_PROVIDER setting instead of
    being hardcoded. Only "ollama" is supported.

    Use this when the caller already knows which model it wants. For
    picking a model based on the question itself, use route_llm() below.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider != "ollama":
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. "
            "Only 'ollama' is supported."
        )

    return _build_ollama(model or settings.OLLAMA_MODEL)


def is_simple_question(question: str) -> bool:
    """Heuristic: a question with few words is considered "simple".

    Word count is a crude proxy for complexity, but it's cheap, has no
    extra dependency, and is good enough to decide between a fast small
    model and a slower general-purpose one. Threshold is configurable
    (ROUTING_WORD_THRESHOLD) since the right cutoff is something you'd
    tune by watching real questions, not derive from theory.
    """
    word_count = len(question.split())
    return word_count <= settings.ROUTING_WORD_THRESHOLD


def route_llm(question: str):
    """Pick a chat model based on the question's complexity.

    Short/simple questions ("what is HTML?") route to OLLAMA_MODEL_FAST
    (qwen3:4b by default) - it loads and answers faster, and is enough
    model for a short factual question. Longer/more complex questions
    route to OLLAMA_MODEL (qwen2.5:7b by default) - more capable, worth
    the extra load/inference time when the question actually needs it.

    Both models are free, open-weight, and run locally via Ollama, and
    both fit in VRAM together on this machine, so switching between just
    these two never evicts the other - only the very first call to each
    pays a real cold-load cost (~3s for the fast model, ~11s for the
    general one); every call after that is warm.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider != "ollama":
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{settings.LLM_PROVIDER}'. "
            "Only 'ollama' is supported."
        )

    model = settings.OLLAMA_MODEL_FAST if is_simple_question(question) else settings.OLLAMA_MODEL
    return _build_ollama(model)


if __name__=="__main__":
    for question in [
        "what is HTML?",
        "Explain in detail how the browser rendering pipeline turns HTML, CSS, and JavaScript into pixels on the screen.",
    ]:
        llm = route_llm(question)
        print(f"[{llm.model}] {question}")
        print(llm.invoke(question).content)
        print()
