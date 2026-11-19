"""Endpoints behind Guide, Quiz, Flashcards, Placement and Progress.

Practice works signed out - nobody should have to make an account to
revise - but only a signed-in student's results are recorded, and
Progress is theirs alone.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import auth
from database.store import (
    create_practice_set, get_guide, get_practice_set, progress, record_result,
    save_guide, save_practice_response,
)
from llm.provider import LLMUnavailableError
from rag.vectorstore import list_indexed_documents
from study.generate import (
    evaluate_interview_answer, generate_flashcards, generate_guide,
    generate_interview_question, generate_quiz,
)

logger = logging.getLogger(__name__)

router = APIRouter()

NOTHING_TO_TEACH = (
    "Could not make practice material from this document - it may be mostly "
    "images, tables or very short pages."
)


def _document(doc_id: str) -> dict:
    match = next((d for d in list_indexed_documents() if d["doc_id"] == doc_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="That document is not in the knowledge base.")
    return match


def _owned_set(set_id: str, kind: str, request: Request) -> dict:
    """A practice set, if the caller may answer it.

    A set made while signed in belongs to that student; someone else
    holding its id gets a 404, the same as a set that never existed.
    """
    practice = get_practice_set(set_id)
    if practice is None or practice["kind"] != kind:
        raise HTTPException(status_code=404, detail="That practice set was not found.")
    if practice["user_id"] is not None:
        user = auth.current_user(request)
        if user is None or user.id != practice["user_id"]:
            raise HTTPException(status_code=404, detail="That practice set was not found.")
    return practice


def _model_down(exc: LLMUnavailableError):
    return HTTPException(status_code=503, detail=str(exc))


# --- guide ------------------------------------------------------------------

@router.get("/study/guide/{doc_id}")
def study_guide(doc_id: str, refresh: bool = False):
    doc = _document(doc_id)
    if not refresh:
        cached = get_guide(doc_id, doc["chunks"])
        if cached:
            return cached
    try:
        guide = generate_guide(doc_id, doc["filename"])
    except LLMUnavailableError as exc:
        raise _model_down(exc)
    if guide is None:
        raise HTTPException(status_code=422, detail=NOTHING_TO_TEACH)
    save_guide(doc_id, doc["chunks"], guide)
    return guide


# --- quiz -------------------------------------------------------------------

class PracticeRequest(BaseModel):
    doc_id: str
    count: int = Field(default=5, ge=1, le=10)


class QuizAnswer(BaseModel):
    index: int = Field(ge=0)
    choice: int = Field(ge=0, le=3)


@router.post("/study/quiz")
def new_quiz(body: PracticeRequest, request: Request):
    doc = _document(body.doc_id)
    user = auth.current_user(request)
    try:
        items = generate_quiz(body.doc_id, body.count)
    except LLMUnavailableError as exc:
        raise _model_down(exc)
    if not items:
        raise HTTPException(status_code=422, detail=NOTHING_TO_TEACH)

    quiz_id = create_practice_set("quiz", doc["doc_id"], doc["filename"], items,
                                  user.id if user else None)
    return {
        "quiz_id": quiz_id,
        "filename": doc["filename"],
        # The answers stay on the server; see PracticeSet.
        "questions": [{"question": i["question"], "options": i["options"]} for i in items],
        "recorded": user is not None,
    }


@router.post("/study/quiz/{quiz_id}/answer")
def answer_quiz(quiz_id: str, body: QuizAnswer, request: Request):
    practice = _owned_set(quiz_id, "quiz", request)
    items = practice["items"]
    if body.index >= len(items):
        raise HTTPException(status_code=400, detail="No such question in this quiz.")

    responses, saved = save_practice_response(quiz_id, body.index, body.choice)
    item = items[body.index]
    chosen = responses[str(body.index)]

    score = sum(1 for i, it in enumerate(items) if responses.get(str(i)) == it["answer"])
    finished = len(responses) == len(items)
    if finished and saved and practice["user_id"] is not None:
        record_result(practice["user_id"], "quiz", practice["doc_id"],
                      practice["filename"], score, len(items))

    return {
        "correct": chosen == item["answer"],
        "chosen": chosen,
        "answer": item["answer"],
        "explanation": item["explanation"],
        "source": item["source"],
        "finished": finished,
        "score": score,
        "total": len(items),
    }


# --- flashcards ---------------------------------------------------------------

class CardReview(BaseModel):
    doc_id: str
    known: int = Field(ge=0)
    total: int = Field(ge=1, le=50)


@router.post("/study/flashcards")
def new_flashcards(body: PracticeRequest):
    doc = _document(body.doc_id)
    try:
        cards = generate_flashcards(body.doc_id, body.count)
    except LLMUnavailableError as exc:
        raise _model_down(exc)
    if not cards:
        raise HTTPException(status_code=422, detail=NOTHING_TO_TEACH)
    return {"filename": doc["filename"], "cards": cards}


@router.post("/study/flashcards/review")
def review_flashcards(body: CardReview, user=Depends(auth.require_user)):
    if body.known > body.total:
        raise HTTPException(status_code=400, detail="Known cannot exceed the number reviewed.")
    doc = _document(body.doc_id)
    record_result(user.id, "flashcards", doc["doc_id"], doc["filename"], body.known, body.total)
    return {"recorded": True}


# --- placement: interview practice ----------------------------------------

class InterviewRequest(BaseModel):
    doc_id: str


class InterviewAnswer(BaseModel):
    answer: str = Field(max_length=4000)


@router.post("/study/interview")
def new_interview_question(body: InterviewRequest, request: Request):
    doc = _document(body.doc_id)
    user = auth.current_user(request)
    try:
        item = generate_interview_question(body.doc_id)
    except LLMUnavailableError as exc:
        raise _model_down(exc)
    if item is None:
        raise HTTPException(status_code=422, detail=NOTHING_TO_TEACH)

    interview_id = create_practice_set("interview", doc["doc_id"], doc["filename"],
                                       [item], user.id if user else None)
    return {
        "interview_id": interview_id,
        "filename": doc["filename"],
        "question": item["question"],
        "source": item["source"],
        "recorded": user is not None,
    }


@router.post("/study/interview/{interview_id}/answer")
def answer_interview(interview_id: str, body: InterviewAnswer, request: Request):
    practice = _owned_set(interview_id, "interview", request)
    item = practice["items"][0]

    existing = practice["responses"].get("0")
    if existing is None:
        try:
            feedback = evaluate_interview_answer(item, body.answer)
        except LLMUnavailableError as exc:
            raise _model_down(exc)
        except RuntimeError:
            raise HTTPException(status_code=502, detail="Could not score that answer. Please try again.")
        responses, saved = save_practice_response(interview_id, 0, feedback)
        if saved and practice["user_id"] is not None:
            record_result(practice["user_id"], "interview", practice["doc_id"],
                          practice["filename"], feedback["score"], 5)
        existing = responses["0"]

    return {**existing, "key_points": item["key_points"], "source": item["source"]}


# --- progress -------------------------------------------------------------------

@router.get("/me/progress")
def my_progress(tz_offset: int = 0, user=Depends(auth.require_user)):
    if not -840 <= tz_offset <= 840:
        tz_offset = 0
    return progress(user.id, tz_offset_minutes=tz_offset)
