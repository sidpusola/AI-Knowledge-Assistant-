"""Thumbs up/down on an answer, and what we do with them.

Stored as JSON Lines - one object per line, opened in append mode and
closed again. That is a deliberate choice over a table:

  * an append is atomic enough for this. Two requests landing together
    produce two lines, never a half-written one, because each write is a
    single short line flushed in one go.
  * it survives a restart, which the in-memory SESSIONS dict does not.
  * it is readable with `tail`, and loadable into pandas in one line when
    you want to look at a week of it.
  * nothing to migrate when the shape changes - an older line simply
    lacks the newer keys.

What it is NOT: concurrent-safe across processes, or queryable at scale.
If this ever runs behind more than one worker, this file becomes a
table. The limitation is real and worth stating rather than discovering.

Why keep it at all: it is the only signal in the whole system that comes
from a student rather than from a metric. Retrieval accuracy and latency
say whether the machinery works; this says whether the answer helped.
"""

import json
import logging
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

# Serialises writes from the threads uvicorn runs sync endpoints on.
# Does nothing across processes - see the module docstring.
_WRITE_LOCK = threading.Lock()

MAX_NOTE = 500
MAX_EXCERPT = 2000


class FeedbackRequest(BaseModel):
    """What the browser posts when someone rates an answer."""

    session_id: str | None = None
    rating: Literal["up", "down"]
    question: str = Field(default="", max_length=1000)
    # Enough of the answer to recognise it later without storing an essay
    # per vote.
    answer_excerpt: str = Field(default="", max_length=MAX_EXCERPT)
    sources: list[str] = Field(default_factory=list, max_length=20)
    note: str | None = Field(default=None, max_length=MAX_NOTE)


class FeedbackSummary(BaseModel):
    total: int
    up: int
    down: int
    # None rather than 0 when nothing has been rated - "no data" and
    # "everyone hated it" should not look the same on a dashboard.
    satisfaction: float | None
    with_sources: int
    without_sources: int


def record_feedback(payload: FeedbackRequest) -> None:
    """Append one rating. Never raises into the request path.

    A failure to record a thumbs-down must not turn into a 500 for the
    student who gave it - the rating is diagnostic, not the product.
    """
    row = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": payload.session_id,
        "rating": payload.rating,
        "question": payload.question,
        "answer_excerpt": payload.answer_excerpt,
        "sources": payload.sources,
        # Whether the answer was grounded is the thing worth correlating
        # a rating against, so it is stored flat rather than inferred.
        "grounded": bool(payload.sources),
        "note": payload.note,
    }

    try:
        path = Path(settings.FEEDBACK_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        logger.warning("[feedback] could not record: %s", exc)


def read_feedback() -> list[dict]:
    """Every rating recorded so far. A malformed line is skipped, not
    fatal - a truncated final line after a hard kill should not make the
    whole history unreadable."""
    path = Path(settings.FEEDBACK_PATH)
    if not path.exists():
        return []

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def feedback_summary() -> FeedbackSummary:
    rows = read_feedback()
    counts = Counter(row.get("rating") for row in rows)
    up, down = counts.get("up", 0), counts.get("down", 0)
    rated = up + down

    return FeedbackSummary(
        total=len(rows),
        up=up,
        down=down,
        satisfaction=(up / rated) if rated else None,
        with_sources=sum(1 for row in rows if row.get("grounded")),
        without_sources=sum(1 for row in rows if not row.get("grounded")),
    )


if __name__ == "__main__":
    summary = feedback_summary()
    print(f"  file        : {settings.FEEDBACK_PATH}")
    print(f"  ratings     : {summary.total}  ({summary.up} up, {summary.down} down)")
    if summary.satisfaction is None:
        print("  satisfaction: no ratings yet")
    else:
        print(f"  satisfaction: {summary.satisfaction:.0%}")
    print(f"  grounded    : {summary.with_sources} with sources, "
          f"{summary.without_sources} without")
