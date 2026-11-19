"""Opening the database, and every query the app makes against it.

Deliberately a thin repository rather than an ORM scattered through the
request handlers: app.py should ask for "this user's sessions", not
build queries. It also means the day SQLite is outgrown, this file is
what gets rewritten and nothing else.

SQLite specifics worth knowing, both handled in _on_connect below:

  * foreign keys are OFF by default in SQLite, per connection. Without
    the pragma, "ON DELETE CASCADE" in the models is decoration and
    deleting a user leaves their conversations behind forever.
  * the default journal is rollback, which makes a reader block a
    writer. WAL lets the student reading their history and the request
    writing the next turn proceed at once.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from config import settings
from database.models import (
    Base, ChatSession, Message, PracticeResult, PracticeSet, StudyGuide, User,
    utcnow,
)

logger = logging.getLogger(__name__)

_IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")

# check_same_thread=False because uvicorn runs sync endpoints on a
# thread pool, so a connection made on one thread is used on another.
# Safe here: each request takes its own session from the factory.
_engine: Engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
)

SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


@event.listens_for(_engine, "connect")
def _on_connect(dbapi_connection, _record):
    if not _IS_SQLITE:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")   # make the cascades real
    cursor.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
    cursor.close()


def init_db() -> None:
    """Create the file and the tables if they aren't there yet.

    create_all only adds what is missing - it never alters an existing
    table. That is fine while the schema only grows; a column that
    changes type or goes away needs a migration, and that is the point
    at which Alembic earns its place.
    """
    if _IS_SQLITE:
        path = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
        path.parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(_engine)
    logger.info("[db] ready at %s", settings.DATABASE_URL)


@contextmanager
def session_scope() -> Iterator[OrmSession]:
    """A unit of work: commits on success, rolls back on any exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --- users -----------------------------------------------------------

def upsert_google_user(sub: str, email: str, email_verified: bool) -> User:
    """Find or create the user behind a Google account.

    Keyed on sub rather than email on purpose - see the note in
    models.py. An existing user's email is refreshed in case they
    changed it on Google's side.
    """
    with session_scope() as session:
        user = session.scalar(select(User).where(User.google_sub == sub))
        if user is None:
            user = User(google_sub=sub, email=email, email_verified=email_verified)
            session.add(user)
            logger.info("[db] new user %s", email)
        else:
            user.email = email
            user.email_verified = email_verified
        user.last_seen_at = utcnow()
        session.flush()
        return user


def get_user(user_id: int) -> User | None:
    with session_scope() as session:
        return session.get(User, user_id)


def complete_profile(user_id: int, display_name: str, dob: date | None) -> User | None:
    """Record the details asked for after the first sign-in."""
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            return None
        user.display_name = display_name.strip()[:80]
        user.date_of_birth = dob
        user.onboarded = True
        session.flush()
        return user


# --- sessions --------------------------------------------------------

def _title_from(question: str) -> str:
    text = " ".join(question.split())
    return text[:117] + "..." if len(text) > 120 else text


def create_session(user_id: int, session_id: str | None = None) -> ChatSession:
    with session_scope() as session:
        chat = ChatSession(id=session_id or str(uuid4()), user_id=user_id)
        session.add(chat)
        session.flush()
        return chat


def get_session(session_id: str, user_id: int) -> ChatSession | None:
    """Scoped to the owner, always.

    Session ids are unguessable, but "hard to guess" is not an access
    control. Every read goes through the user_id so a leaked or shared
    id still cannot open someone else's conversation.
    """
    with session_scope() as session:
        return session.scalar(
            select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user_id
            )
        )


def list_sessions(user_id: int, limit: int = 50) -> list[dict]:
    """A user's conversations, newest first, for the sidebar."""
    with session_scope() as session:
        rows = session.scalars(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "title": row.title or "New conversation",
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
                "messages": len(row.messages),
            }
            for row in rows
        ]


def delete_session(session_id: str, user_id: int) -> bool:
    with session_scope() as session:
        chat = session.scalar(
            select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user_id
            )
        )
        if chat is None:
            return False
        session.delete(chat)   # messages go with it, via the cascade
        return True


# --- messages --------------------------------------------------------

def record_turn(
    session_id: str,
    user_id: int,
    question: str,
    answer: str,
    sources: list[str] | None = None,
    model: str | None = None,
) -> None:
    """Persist one exchange, creating the conversation if it is new.

    Both messages and the session's timestamp move together in one
    transaction, so a crash between them cannot leave a question stored
    without its answer.
    """
    with session_scope() as session:
        chat = session.get(ChatSession, session_id)
        if chat is None:
            chat = ChatSession(id=session_id, user_id=user_id)
            session.add(chat)
        elif chat.user_id != user_id:
            # Someone else's conversation id. Refuse rather than write
            # into it.
            logger.warning("[db] refused cross-user write to session %s", session_id)
            return

        if not chat.title:
            chat.title = _title_from(question)

        session.add(Message(session_id=session_id, role="user", content=question))
        session.add(
            Message(
                session_id=session_id,
                role="assistant",
                content=answer,
                sources="\n".join(sources) if sources else None,
                model=model,
            )
        )
        chat.updated_at = utcnow()


def load_history(session_id: str, user_id: int, turns: int | None = None) -> list:
    """The recent conversation as LangChain messages, oldest first.

    Only the last `turns` exchanges are returned: the whole history is
    kept, but replaying all of it into the prompt would grow the context
    without bound and cost more on every question.
    """
    turns = turns or settings.HISTORY_TURNS
    with session_scope() as session:
        chat = session.scalar(
            select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user_id
            )
        )
        if chat is None:
            return []

        recent = chat.messages[-(turns * 2):]
        return [
            HumanMessage(content=m.content) if m.role == "user"
            else AIMessage(content=m.content)
            for m in recent
        ]


def load_transcript(session_id: str, user_id: int) -> list[dict]:
    """The full conversation, for showing a resumed session in the UI."""
    with session_scope() as session:
        chat = session.scalar(
            select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user_id
            )
        )
        if chat is None:
            return []
        return [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources.split("\n") if m.sources else [],
                "created_at": m.created_at.isoformat(),
            }
            for m in chat.messages
        ]


# --- reporting -------------------------------------------------------

def user_stats(user_id: int) -> dict:
    """Per-student activity, for the NGO's own reporting."""
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            return {}

        sessions = user.sessions
        questions = sum(
            1 for s in sessions for m in s.messages if m.role == "user"
        )
        first = min((s.created_at for s in sessions), default=None)
        last = max((s.updated_at for s in sessions), default=None)

        return {
            "user_id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "sessions": len(sessions),
            "questions": questions,
            "first_seen": (first or user.created_at).isoformat(),
            "last_seen": (last or user.last_seen_at).isoformat(),
        }


# --- practice --------------------------------------------------------

def create_practice_set(kind: str, doc_id: str, filename: str, items: list[dict],
                        user_id: int | None) -> str:
    with session_scope() as session:
        practice = PracticeSet(
            id=str(uuid4()), kind=kind, user_id=user_id, doc_id=doc_id,
            filename=filename, items=json.dumps(items),
        )
        session.add(practice)
        return practice.id


def get_practice_set(set_id: str) -> dict | None:
    with session_scope() as session:
        practice = session.get(PracticeSet, set_id)
        if practice is None:
            return None
        return {
            "id": practice.id,
            "kind": practice.kind,
            "user_id": practice.user_id,
            "doc_id": practice.doc_id,
            "filename": practice.filename,
            "items": json.loads(practice.items),
            "responses": json.loads(practice.responses or "{}"),
        }


def save_practice_response(set_id: str, index: int, response) -> tuple[dict, bool]:
    """Record the response to one item, unless it was already answered.

    First answer wins. Re-submitting an answered question returns what
    was recorded rather than overwriting it, so a score can't be raised
    by answering until right. The flag says whether this call saved it.
    """
    with session_scope() as session:
        practice = session.get(PracticeSet, set_id)
        responses = json.loads(practice.responses or "{}")
        key = str(index)
        if key in responses:
            return responses, False
        responses[key] = response
        practice.responses = json.dumps(responses)
        return responses, True


def record_result(user_id: int, kind: str, doc_id: str, filename: str,
                  score: float, out_of: float) -> None:
    with session_scope() as session:
        session.add(PracticeResult(
            user_id=user_id, kind=kind, doc_id=doc_id, filename=filename,
            score=score, out_of=out_of,
        ))


def get_guide(doc_id: str, chunks: int) -> dict | None:
    """The cached guide, if it was made from the document as it is now."""
    with session_scope() as session:
        guide = session.get(StudyGuide, doc_id)
        if guide is None or guide.chunks != chunks:
            return None
        return json.loads(guide.content)


def save_guide(doc_id: str, chunks: int, content: dict) -> None:
    with session_scope() as session:
        guide = session.get(StudyGuide, doc_id)
        if guide is None:
            session.add(StudyGuide(doc_id=doc_id, chunks=chunks, content=json.dumps(content)))
        else:
            guide.chunks = chunks
            guide.content = json.dumps(content)
            guide.created_at = utcnow()


def _ratio(score: float, out_of: float) -> float | None:
    return round(score / out_of, 3) if out_of else None


def progress(user_id: int, tz_offset_minutes: int = 0, days: int = 14) -> dict:
    """Everything the Progress screen shows, from real records only.

    tz_offset_minutes is the browser's getTimezoneOffset(): minutes
    *behind* UTC, so India is -330. Without it a question asked at 1am
    in Delhi lands on yesterday's bar.
    """
    shift = timedelta(minutes=-tz_offset_minutes)

    with session_scope() as session:
        results = session.scalars(
            select(PracticeResult)
            .where(PracticeResult.user_id == user_id)
            .order_by(PracticeResult.created_at)
        ).all()
        asked = session.scalars(
            select(Message.created_at)
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(ChatSession.user_id == user_id, Message.role == "user")
        ).all()
        conversations = session.scalar(
            select(func.count()).select_from(ChatSession)
            .where(ChatSession.user_id == user_id)
        ) or 0

        def totals(kind: str) -> dict:
            rows = [r for r in results if r.kind == kind]
            score = sum(r.score for r in rows)
            out_of = sum(r.out_of for r in rows)
            return {"count": len(rows), "score": score, "out_of": out_of,
                    "ratio": _ratio(score, out_of)}

        per_doc: dict[str, dict] = {}
        for row in results:
            doc = per_doc.setdefault(row.doc_id, {
                "doc_id": row.doc_id, "filename": row.filename,
                "quiz_best": None, "quiz_last": None, "quizzes": 0,
                "cards_known": None, "interview_avg": None, "interviews": 0,
                "last_active": None,
            })
            ratio = _ratio(row.score, row.out_of)
            if row.kind == "quiz":
                doc["quizzes"] += 1
                doc["quiz_last"] = ratio
                doc["quiz_best"] = max(doc["quiz_best"] or 0, ratio or 0)
            elif row.kind == "flashcards":
                doc["cards_known"] = ratio
            elif row.kind == "interview":
                total = (doc["interview_avg"] or 0) * doc["interviews"] + row.score
                doc["interviews"] += 1
                doc["interview_avg"] = round(total / doc["interviews"], 2)
            doc["last_active"] = row.created_at.isoformat()

        today = (utcnow() + shift).date()
        window = [today - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
        activity = {day: {"questions": 0, "practice": 0} for day in window}
        for when in asked:
            day = (when + shift).date()
            if day in activity:
                activity[day]["questions"] += 1
        for row in results:
            day = (row.created_at + shift).date()
            if day in activity:
                activity[day]["practice"] += 1

        return {
            "questions_asked": len(asked),
            "conversations": conversations,
            "quiz": totals("quiz"),
            "flashcards": totals("flashcards"),
            "interview": totals("interview"),
            "documents": sorted(per_doc.values(),
                                key=lambda d: d["last_active"] or "", reverse=True),
            "activity": [{"date": day.isoformat(), **counts}
                         for day, counts in activity.items()],
            "recent": [
                {"kind": r.kind, "filename": r.filename, "score": r.score,
                 "out_of": r.out_of, "created_at": r.created_at.isoformat()}
                for r in reversed(results[-10:])
            ],
        }


if __name__ == "__main__":
    init_db()
    print(f"  database : {settings.DATABASE_URL}")
    with session_scope() as session:
        for label, model in (("users", User), ("sessions", ChatSession),
                             ("messages", Message)):
            count = session.scalar(select(func.count()).select_from(model))
            print(f"  {label:9}: {count}")
