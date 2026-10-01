"""
Lightweight request instrumentation - Phase 1 of monitoring.

No extra services to run: timings go to the console and into a small
in-memory ring buffer that the /stats endpoint exposes as JSON.

The measurement points here (retrieval time, LLM time, time-to-first-token,
total) are deliberately the same ones Prometheus histograms would wrap
later, so adding Prometheus + Grafana becomes mostly configuration rather
than re-instrumenting the code from scratch.

Caveat, same as the session store in app.py: this is in-memory and
per-process. It resets on restart and isn't shared across workers.
"""

import logging
from collections import deque
from dataclasses import asdict, dataclass
from threading import Lock
from time import perf_counter

MAX_RECENT = 50

logger = logging.getLogger("ai_agent")

# FastAPI runs sync endpoints in a threadpool, so several requests can be
# recording at once - the deque append is atomic but the stats read isn't.
_lock = Lock()
_recent: deque = deque(maxlen=MAX_RECENT)
_request_counter = 0


def _ensure_handler() -> None:
    """Attach our own handler instead of relying on the root logger.

    Uvicorn configures its own loggers but leaves the root logger alone, so
    an un-handled INFO record would never be printed. Guarded so uvicorn's
    --reload doesn't stack duplicate handlers.
    """
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


_ensure_handler()


class Timer:
    """Context manager that records how long its block took.

        with Timer() as t:
            do_work()
        t.seconds
    """

    def __enter__(self) -> "Timer":
        self.seconds = 0.0
        self._start = perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.seconds = perf_counter() - self._start


@dataclass
class RequestRecord:
    id: int
    question: str
    model: str
    retrieval_seconds: float
    llm_seconds: float
    total_seconds: float
    streamed: bool
    time_to_first_token: float | None = None
    error: str | None = None


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.2f}s"


def record_request(
    question: str,
    model: str,
    retrieval_seconds: float,
    llm_seconds: float,
    total_seconds: float,
    streamed: bool = False,
    time_to_first_token: float | None = None,
    error: str | None = None,
) -> RequestRecord:
    """Log one request's timings and keep it in the recent-requests buffer."""
    global _request_counter

    with _lock:
        _request_counter += 1
        record = RequestRecord(
            id=_request_counter,
            question=question,
            model=model,
            retrieval_seconds=round(retrieval_seconds, 4),
            llm_seconds=round(llm_seconds, 4),
            total_seconds=round(total_seconds, 4),
            streamed=streamed,
            time_to_first_token=(
                round(time_to_first_token, 4) if time_to_first_token is not None else None
            ),
            error=error,
        )
        _recent.append(record)

    label = "stream" if streamed else "ask"
    parts = [f"retrieval {_fmt(retrieval_seconds)}"]
    if time_to_first_token is not None:
        parts.append(f"ttft {_fmt(time_to_first_token)}")
    parts.append(f"llm {_fmt(llm_seconds)}")
    parts.append(f"total {_fmt(total_seconds)}")
    if error:
        parts.append(f"ERROR {error}")

    preview = question if len(question) <= 60 else question[:57] + "..."
    logger.info("[%s #%d] %s | %s | %r", label, record.id, model, " | ".join(parts), preview)

    return record


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Fine for a 50-entry buffer; don't read too
    much into it with only a handful of samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return round(ordered[index], 4)


def get_stats() -> dict:
    """Aggregates over the recent-requests buffer, for the /stats endpoint."""
    with _lock:
        records = list(_recent)
        total_seen = _request_counter

    if not records:
        return {
            "total_requests": total_seen,
            "recent_sample_size": 0,
            "recent": [],
        }

    totals = [r.total_seconds for r in records]
    retrievals = [r.retrieval_seconds for r in records]
    llms = [r.llm_seconds for r in records]
    ttfts = [r.time_to_first_token for r in records if r.time_to_first_token is not None]
    errors = [r for r in records if r.error]

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    by_model: dict[str, int] = {}
    for r in records:
        by_model[r.model] = by_model.get(r.model, 0) + 1

    return {
        "total_requests": total_seen,
        "recent_sample_size": len(records),
        "errors_in_sample": len(errors),
        "requests_by_model": by_model,
        "latency_seconds": {
            "avg_total": avg(totals),
            "p95_total": _percentile(totals, 0.95),
            "avg_retrieval": avg(retrievals),
            "avg_llm": avg(llms),
            "avg_time_to_first_token": avg(ttfts),
        },
        # newest first - easier to eyeball
        "recent": [asdict(r) for r in reversed(records)],
    }
