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


def safe_stream(chain, chain_input):
    
    try:
        yield from chain.stream(chain_input)
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


_THINK_END_TAG = "</think>"


def strip_thinking(text: str) -> str:  #for leaked thinking
    
    if _THINK_END_TAG in text:
        return text.split(_THINK_END_TAG, 1)[1].lstrip()
    return text


def _build_ollama(model: str) -> ChatOllama:
    
    kwargs = {}
    if model in settings.REASONING_MODELS:
        kwargs["reasoning"] = True

    return ChatOllama(
        model=model,
        temperature=settings.OLLAMA_TEMPERATURE,
        **kwargs,
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


# Question shapes that are nearly always a straight factual lookup.
# Matched at the start of the question only - "what is X" opens a lookup,
# but "tell me what is wrong with this design" is not one.
_SIMPLE_OPENERS = (
    "what is", "what are", "what does", "what do",
    "define", "definition of",
    "who is", "who was", "who are",
    "when is", "when was", "when did",
    "where is", "where are",
    "list the", "name the",
)

# Signals that real reasoning is needed no matter how short the question
# is. These override the openers above: "what is the difference between
# X and Y" opens like a lookup but isn't one.
_COMPLEX_SIGNALS = (
    "why", "how does", "how do ", "how would", "how can", "how should",
    "explain", "compare", "contrast", "difference between",
    "derive", "prove", "analyse", "analyze", "evaluate", "justify",
    "design", "debug", "optimise", "optimize", "implement", "refactor",
    "trade-off", "tradeoff", "pros and cons", "advantages and disadvantages",
    "step by step", "in detail", "walk me through", "in your own words",
)


def is_simple_question(question: str) -> bool:
    """Decide whether a question is a simple factual lookup.

    Deliberately conservative: the default is the capable model, and a
    question only gets diverted to the small one when it clearly looks
    like a lookup. That asymmetry is intentional, because the two kinds
    of mistake do not cost the same - measured on this machine, routing
    an easy question to the big model costs about 0.9s, while routing a
    hard one to the small model costs a noticeably worse answer.

    Word count alone used to decide this, which got obvious cases
    backwards: "Prove that P != NP" is 5 words but hard, while a wordy
    "can you tell me what HTML stands for" is long but trivial. Length
    measures verbosity, not difficulty, so it's now only a backstop.
    """
    text = question.lower().strip()

    # A reasoning signal anywhere wins, however short the question is.
    if any(signal in text for signal in _COMPLEX_SIGNALS):
        return False

    # More than one question mark means multiple things are being asked.
    if text.count("?") > 1:
        return False

    # Length is kept as a backstop only - a rambling question is more
    # likely to be involved, and misjudging it merely costs ~0.9s.
    if len(text.split()) > settings.ROUTING_WORD_THRESHOLD:
        return False

    # Nothing above ruled it out, so divert to the small model only if it
    # actually opens like a lookup. Anything unrecognised stays on the
    # capable model.
    return text.startswith(_SIMPLE_OPENERS)


def route_llm(question: str):
    
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
