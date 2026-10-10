"""What a user, a conversation and a turn look like on disk.

Three tables, because that is what "track each user's history and
sessions over time" actually needs:

    users      one row per student, keyed by their Google account
    sessions   one conversation, belonging to a user
    messages   one turn within a conversation

Until now conversations lived in a dict in app.py and were lost on every
restart. That was fine while there were no users; it is not fine once a
student is meant to come back tomorrow and find their work.

Identity comes from Google, so there is no password column here and
never should be. google_sub - not email - is the key: Google documents
sub as the stable, never-reused identifier for an account, while an
email address can be changed or reassigned. Email is stored because it
is useful to staff, not because anything depends on it.

Personal data is kept deliberately thin: a display name and a date of
birth, because an NGO reporting on who it reaches needs ages. No
addresses, no phone numbers, nothing that isn't used.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware UTC. SQLite has no native datetime type and will
    happily store whatever it is given, so the aware value is created
    here rather than left to the database."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Google's stable account identifier. Unique and never reused, which
    # is why it - and not the email address - is what a row is keyed on.
    google_sub: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    email_verified: Mapped[bool] = mapped_column(default=False)

    # Supplied by the student after their first sign-in, not by Google:
    # Google's name field is whatever they put on their account, which
    # is often not what they want to be called here.
    display_name: Mapped[str | None] = mapped_column(String(80), default=None)
    date_of_birth: Mapped[date | None] = mapped_column(Date, default=None)

    # Separate from "has a display_name" so the onboarding form can be
    # completed, skipped or revisited without the app having to infer
    # intent from empty fields.
    onboarded: Mapped[bool] = mapped_column(default=False)

    # Staff can see every document and delete them; students cannot.
    role: Mapped[str] = mapped_column(String(16), default="student")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="ChatSession.updated_at.desc()",
    )

    def __repr__(self) -> str:
        return f"<User {self.id} {self.email}>"


class ChatSession(Base):
    """One conversation. Named ChatSession rather than Session because
    SQLAlchemy's own Session is imported in every module that touches
    this one, and two things called Session is a bug waiting to
    happen."""

    __tablename__ = "sessions"

    # A UUID string, not an integer: the browser creates and holds this
    # id, and a guessable sequential one would let anybody walk other
    # people's conversations by changing a number in a URL.
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    # Taken from the opening question, so a returning student recognises
    # the conversation in a list.
    title: Mapped[str | None] = mapped_column(String(120), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )

    def __repr__(self) -> str:
        return f"<ChatSession {self.id} user={self.user_id}>"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )

    role: Mapped[str] = mapped_column(String(16))          # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)

    # Citations as they were shown, newline separated. Stored on the
    # message rather than recomputed, because the index changes: a
    # document deleted next week should not silently rewrite what a
    # student was told today.
    sources: Mapped[str | None] = mapped_column(Text, default=None)

    # Which model answered, for the per-user view and for working out
    # later whether the fast model was the wrong choice somewhere.
    model: Mapped[str | None] = mapped_column(String(64), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message {self.id} {self.role}>"


# The query behind the session list - "this user's conversations, newest
# first" - runs on every page load, so it gets an index rather than a
# scan per visit.
Index("ix_sessions_user_updated", ChatSession.user_id, ChatSession.updated_at.desc())
