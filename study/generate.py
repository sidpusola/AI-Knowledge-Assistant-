"""Practice material made from a student's own documents.

Quizzes, flashcards, study guides and interview questions, each built
from passages of one indexed document, so everything a student
practises is something their course actually teaches - and each item
carries the page it came from.

The model writes; code decides. Every property that can be checked is
checked here rather than requested in the prompt, because a local 7B
follows a formatting rule most of the time, and "most of the time" is a
quiz with a broken question in it:

  * where the right answer sits among the options is chosen in code. The
    model writes the correct answer and three wrong ones separately, so
    it never has to count, and the answer is not always option A.
  * a question that refers to "the passage" is dropped. The student
    never sees the passage.
  * "all of the above" / "none of the above" are dropped - they stop
    meaning anything once the options are shuffled.
  * an interview answer too short to judge is scored without a model
    call at all.
"""

from __future__ import annotations

import random
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from chains.summarize import summarize_text
from llm.provider import LLMUnavailableError, get_llm, safe_invoke
from rag.metadata import describe
from rag.vectorstore import load_vectorstore

# A passage worth teaching from. Below these it is usually a heading, a
# table-of-contents line, a figure caption or an index page.
MIN_PASSAGE_CHARS = 280
MIN_PASSAGE_WORDS = 40
MIN_LETTER_RATIO = 0.6

GUIDE_PASSAGES = 12
GUIDE_CHARS = 7000

_MENTIONS_SOURCE = re.compile(
    r"\b(?:the|this|above|given|following) (?:passage|excerpt|extract|paragraph"
    # "the text editor" is a computing question, not a reference to the source.
    r"|text(?!\s+(?:editor|box|field|file|message|area|string|format|mode)))\b"
    # Not bare "the context": "in the context of a study" is a fine question.
    r"|\bgiven statements?\b|\b(?:context|information) provided\b"
    r"|\bprovided (?:context|information)\b",
    re.I,
)
_CATCH_ALL = re.compile(r"\b(?:all|none|both) of (?:the )?(?:above|these)\b", re.I)

# Measured on a recruitment handout: 5 of 6 questions opened "According
# to the passage, ...", and every one was dropped - a whole quiz of zero.
# The opening clause is removable and the question stands without it, so
# it is cut rather than the question thrown away. Only a comma- or
# colon-ended clause, so "In the text editor, ..." is left alone; a
# reference anywhere else in the sentence is still rejected.
_SOURCE_PREAMBLE = re.compile(
    r"^(?:according to|based on|as (?:stated|mentioned|described|explained) in|from|in)"
    r" (?:the|this) (?:passage|text|excerpt|extract|paragraph)\s*[,:]\s*",
    re.I,
)
# The same clause at the end - measured as half of all rejections once the
# opening one was handled: "What are stereotypes according to the
# passage?". Only directly before the closing question mark.
_SOURCE_TAIL = re.compile(
    r",?\s*(?:as (?:stated|mentioned|described|explained|outlined|given) in"
    r"|according to|based on|in) (?:the|this) (?:passage|text|excerpt|extract|paragraph)"
    r"\s*(?=\?$)",
    re.I,
)
# A definition that hedges is the model guessing at something the material
# never explains - seen as "MSB: most likely an abbreviation for
# 'Microphone System Browser' or a similar tool". A student would learn
# the guess, so such a term or card is dropped.
_HEDGE = re.compile(
    r"\b(?:most likely|probably|possibly|presumably|may refer to|might (?:be|refer)"
    r"|or (?:a )?similar|not (?:clearly )?(?:specified|explained|defined|stated))\b",
    re.I,
)
# In an explanation shown beside its page citation, "the passage" is fine
# to keep in meaning - it just needs to name something the student has.
_PASSAGE_WORD = re.compile(r"\b(?:the|this) (?:passage|excerpt|extract|paragraph)\b", re.I)
# Feedback is read by the student it is about.
_THIRD_PERSON = [
    (re.compile(r"\bthe student's\b", re.I), "your"),
    (re.compile(r"\bthe student (?:is)\b", re.I), "you are"),
    (re.compile(r"\bthe student (?:was)\b", re.I), "you were"),
    (re.compile(r"\bthe student (?:has)\b", re.I), "you have"),
    (re.compile(r"\bthe student (?:does)\b", re.I), "you do"),
    (re.compile(r"\bthe student\b", re.I), "you"),
]
_SENTENCE_START = re.compile(r"(^|[.!?]\s+)(you|your)\b")
# The model sometimes writes its own "A) ... B) ..." list into the question
# and letters its options. Position is decided here, so both go.
_INLINE_OPTIONS = re.compile(r"\n\s*\(?A[\).:]\s")
_OPTION_LETTER = re.compile(r"^\(?[A-Da-d][\).:]\s+")


# --- passages -----------------------------------------------------------

def _passages(doc_id: str) -> list[Document]:
    records = load_vectorstore().get(where={"doc_id": doc_id})
    return [
        Document(page_content=text or "", metadata=meta or {})
        for text, meta in zip(records.get("documents") or [],
                              records.get("metadatas") or [])
    ]


def _is_teachable(text: str) -> bool:
    text = text.strip()
    if len(text) < MIN_PASSAGE_CHARS or len(text.split()) < MIN_PASSAGE_WORDS:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters / len(text) >= MIN_LETTER_RATIO


def _page(doc: Document) -> int:
    page = doc.metadata.get("page")
    return int(page) if isinstance(page, (int, float)) else 0


def teachable_passages(doc_id: str) -> list[Document]:
    docs = [d for d in _passages(doc_id) if _is_teachable(d.page_content)]
    docs.sort(key=_page)
    return docs


def sample_passages(doc_id: str, n: int) -> list[Document]:
    """A random draw across the document, in page order.

    Random so a second quiz on the same handout asks different things;
    drawn from the whole document so a 1850-passage textbook is not
    quizzed only on its introduction.
    """
    docs = teachable_passages(doc_id)
    chosen = random.sample(docs, min(n, len(docs)))
    chosen.sort(key=_page)
    return chosen


def spread_passages(doc_id: str, n: int) -> list[Document]:
    """Evenly spaced passages from start to end - for a guide, which
    should cover the document, and should come out the same each time."""
    docs = teachable_passages(doc_id)
    if len(docs) <= n:
        return docs
    step = len(docs) / n
    return [docs[int(i * step)] for i in range(n)]


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _standalone(text: str | None) -> str:
    """Text with an opening "According to the passage," or a closing
    "...according to the passage?" removed."""
    text = _SOURCE_TAIL.sub("", _SOURCE_PREAMBLE.sub("", _clean(text)))
    return text[:1].upper() + text[1:]


def _option(text: str | None) -> str:
    return _OPTION_LETTER.sub("", _clean(text))


def _capitalise_starts(text: str) -> str:
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).capitalize(), text)


def _cite_material(text: str | None) -> str:
    return _capitalise_starts(_PASSAGE_WORD.sub("your material", _clean(text)))


def _to_student(text: str | None) -> str:
    text = _clean(text)
    for pattern, replacement in _THIRD_PERSON:
        text = pattern.sub(replacement, text)
    return _capitalise_starts(text)


def _invoke(chain, inputs: dict):
    """One structured call. A malformed reply costs this item, not the
    whole set - only an unreachable model is worth failing the request."""
    try:
        return safe_invoke(chain, inputs)
    except LLMUnavailableError:
        raise
    except Exception:
        return None


# --- quiz ------------------------------------------------------------------

class _QuizItem(BaseModel):
    question: str = Field(description="The question, answerable without seeing the passage")
    correct_answer: str = Field(description="The right answer, short")
    wrong_answers: list[str] = Field(
        description="Three plausible but wrong answers, each similar in length and style to the right one"
    )
    explanation: str = Field(description="One or two sentences on why the right answer is right")


_QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write one multiple-choice question for a student on a vocational "
     "course, using only the course passage you are given.\n\n"
     "Ask about something the passage states that is worth knowing - a "
     "definition, a requirement, a step, a reason, a rule. Not a page "
     "number, a heading or a name in passing.\n\n"
     "The student will not see the passage, so the question must make "
     "sense on its own."),
    ("human", "Course passage:\n\n{passage}"),
])


def _quiz_item(passage: Document) -> dict | None:
    chain = _QUIZ_PROMPT | get_llm().with_structured_output(_QuizItem)
    raw = _invoke(chain, {"passage": passage.page_content})
    if raw is None:
        return None

    question = _standalone(_INLINE_OPTIONS.split(raw.question or "")[0])
    correct = _option(raw.correct_answer)
    if not question or not correct or _MENTIONS_SOURCE.search(question):
        return None

    seen = {correct.lower()}
    wrong = []
    for option in raw.wrong_answers or []:
        option = _option(option)
        if option and option.lower() not in seen and not _CATCH_ALL.search(option):
            seen.add(option.lower())
            wrong.append(option)
    if len(wrong) < 3 or _CATCH_ALL.search(correct):
        return None

    options = [correct, *wrong[:3]]
    random.shuffle(options)
    return {
        "question": question,
        "options": options,
        "answer": options.index(correct),
        "explanation": _cite_material(raw.explanation),
        "source": describe(passage),
    }


def generate_quiz(doc_id: str, count: int) -> list[dict]:
    """Up to `count` questions. Draws twice as many passages as needed,
    since some will not survive the checks above."""
    items = []
    for passage in sample_passages(doc_id, count * 2):
        item = _quiz_item(passage)
        if item:
            items.append(item)
        if len(items) == count:
            break
    return items


# --- flashcards --------------------------------------------------------------

class _Card(BaseModel):
    front: str = Field(description="A key term, or a short question")
    back: str = Field(description="The definition or answer, in one or two sentences")


_CARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write one revision flashcard from a course passage. The front is "
     "a key term or a short question from the passage; the back is its "
     "meaning or answer in one or two plain sentences. The student will "
     "not see the passage, so both sides must make sense on their own."),
    ("human", "Course passage:\n\n{passage}"),
])


def _card(passage: Document) -> dict | None:
    chain = _CARD_PROMPT | get_llm().with_structured_output(_Card)
    raw = _invoke(chain, {"passage": passage.page_content})
    if raw is None:
        return None
    front, back = _standalone(raw.front), _standalone(raw.back)
    if not front or not back or len(front) > 200 or len(back) > 500:
        return None
    if _MENTIONS_SOURCE.search(front) or _MENTIONS_SOURCE.search(back) or _HEDGE.search(back):
        return None
    return {"front": front, "back": back, "source": describe(passage)}


def generate_flashcards(doc_id: str, count: int) -> list[dict]:
    cards, fronts = [], set()
    for passage in sample_passages(doc_id, count * 2):
        card = _card(passage)
        if card and card["front"].lower() not in fronts:
            fronts.add(card["front"].lower())
            cards.append(card)
        if len(cards) == count:
            break
    return cards


# --- study guide ---------------------------------------------------------------

class _Term(BaseModel):
    term: str
    meaning: str = Field(description="What it means, in one plain sentence")


class _Terms(BaseModel):
    terms: list[_Term] = Field(description="Five to eight key terms a student must know")


_TERMS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "From these course passages, list the key terms a student must know, "
     "each with what it means in one plain sentence. Use only terms the "
     "passages actually explain."),
    ("human", "{text}"),
])


def generate_guide(doc_id: str, filename: str) -> dict | None:
    passages = spread_passages(doc_id, GUIDE_PASSAGES)
    if not passages:
        return None

    text = "\n\n".join(p.page_content.strip() for p in passages)[:GUIDE_CHARS]
    try:
        summary = summarize_text(text)
    except LLMUnavailableError:
        raise
    except Exception:
        summary = None

    terms_raw = _invoke(_TERMS_PROMPT | get_llm().with_structured_output(_Terms),
                        {"text": text})
    terms, seen = [], set()
    for item in (terms_raw.terms if terms_raw else []):
        term, meaning = _clean(item.term), _clean(item.meaning)
        if (term and meaning and len(term) <= 60 and term.lower() not in seen
                and not _HEDGE.search(meaning)):
            seen.add(term.lower())
            terms.append({"term": term, "meaning": meaning})

    overview = _clean(summary.summary) if summary else ""
    key_points = [_clean(p) for p in (summary.key_points if summary else []) if _clean(p)]
    if not overview and not terms:
        return None

    sources = []
    for passage in passages:
        label = describe(passage)
        if label not in sources:
            sources.append(label)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "overview": overview,
        "key_points": key_points,
        "key_terms": terms[:8],
        "sources": sources,
    }


# --- interview practice ------------------------------------------------------

class _InterviewQuestion(BaseModel):
    question: str = Field(
        description="A question an interviewer could ask a candidate for an entry-level job"
    )
    key_points: list[str] = Field(
        description="Two to four points a strong answer would cover, taken from the passage"
    )
    example_answer: str = Field(
        description="How a strong candidate would answer, speaking to the interviewer in the "
                    "first person, in three or four sentences"
    )


# The example answer is written here, with the question, rather than with
# the feedback. Written alongside feedback it came out as more coaching -
# "Hey there! Your answer is a good start...", "let's dive into..." - on
# most runs, because by then the model is reacting to the student.
_INTERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are preparing a student on a vocational course for a job "
     "interview. From the course passage, write one question an "
     "interviewer for an entry-level role could realistically ask, which "
     "this passage prepares the candidate to answer. Ask it the way an "
     "interviewer would speak - not as an exam question - and never "
     "mention the passage. Then list the points a strong answer covers, "
     "and write the answer a strong candidate would give."),
    ("human", "Course passage:\n\n{passage}"),
])


def generate_interview_question(doc_id: str) -> dict | None:
    for passage in sample_passages(doc_id, 4):
        chain = _INTERVIEW_PROMPT | get_llm().with_structured_output(_InterviewQuestion)
        raw = _invoke(chain, {"passage": passage.page_content})
        if raw is None:
            continue
        question = _standalone(raw.question)
        points = [_clean(p) for p in raw.key_points or [] if _clean(p)]
        if question and points and not _MENTIONS_SOURCE.search(question):
            return {
                "question": question,
                "key_points": points[:4],
                "example_answer": _example_answer(raw.example_answer),
                "passage": passage.page_content.strip(),
                "source": describe(passage),
            }
    return None


# The score is worked out here from a judgement per point, not asked for.
# Asked directly, the model gave the same answer 3/5 on one run and 5/5 on
# the next - the 5/5 alongside "you missed the specific application",
# which is the question's whole point. Judging each point separately is
# a smaller, steadier question, and the score then cannot disagree with
# the points shown beside it.
_COVERAGE_WEIGHT = {"covered": 1.0, "partly": 0.5, "missing": 0.0}


class _Feedback(BaseModel):
    coverage: list[str] = Field(
        description="One word per numbered point, in order: covered, partly or missing"
    )
    strengths: list[str] = Field(description="What the answer did well, up to three short points")
    improvements: list[str] = Field(description="What would make it stronger, up to three short points")


_FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly interview coach for a student on a vocational "
     "course. Judge their spoken-style answer against the course material "
     "and the numbered points a strong answer covers. For each point, say "
     "whether the answer covers it, partly covers it, or misses it. Be "
     "encouraging but honest. Speak to the student directly, as \"you\"."),
    ("human",
     "Interview question: {question}\n\n"
     "Points a strong answer covers:\n{key_points}\n\n"
     "Course material:\n{passage}\n\n"
     "The student's answer:\n{answer}"),
])

_CHATTY_OPENING = re.compile(
    r"^(?:certainly|sure|of course|absolutely|great question|here(?:'s| is) a strong answer)[!.,:]?\s*",
    re.I,
)
# An example answer is what the candidate says to the interviewer. One that
# talks about the student's own attempt is more coaching in the wrong box -
# seen as "Hey there! Your answer is a good start, but let's dive deeper".
_COACHING = re.compile(
    r"\byour answer\b|\byou (?:mentioned|said|wrote|stated|described)\b|^(?:hey|hi|hello)\b",
    re.I,
)


def _example_answer(text: str | None) -> str:
    text = _CHATTY_OPENING.sub("", _clean(text))
    if _COACHING.search(text):
        return ""
    return text[:1].upper() + text[1:]


def _verdict(word: str) -> str:
    word = (word or "").strip().lower()
    if word.startswith("cover") or word in ("yes", "full", "fully"):
        return "covered"
    if word.startswith("part"):
        return "partly"
    return "missing"


def score_from_coverage(coverage: list[str]) -> int:
    """1 when every point is missed, 5 when every point is covered."""
    if not coverage:
        return 1
    share = sum(_COVERAGE_WEIGHT[v] for v in coverage) / len(coverage)
    return 1 + int(4 * share + 0.5)

MIN_ANSWER_WORDS = 5


def evaluate_interview_answer(item: dict, answer: str) -> dict:
    answer = (answer or "").strip()
    points = item["key_points"]
    if len(answer.split()) < MIN_ANSWER_WORDS:
        # Nothing to judge, and no model needed to say so.
        return {
            "score": 1,
            "coverage": ["missing"] * len(points),
            "strengths": [],
            "improvements": ["Give a full answer - a few sentences that cover the points below."],
            "model_answer": item.get("example_answer", ""),
        }

    chain = _FEEDBACK_PROMPT | get_llm().with_structured_output(_Feedback)
    raw = _invoke(chain, {
        "question": item["question"],
        "key_points": "\n".join(f"{n}. {p}" for n, p in enumerate(points, start=1)),
        "passage": item["passage"],
        "answer": answer[:4000],
    })
    if raw is None:
        raise RuntimeError("feedback could not be produced")

    # A short or long list is lined up with the points rather than
    # trusted: a point the model skipped counts as missed.
    verdicts = [_verdict(v) for v in (raw.coverage or [])][:len(points)]
    verdicts += ["missing"] * (len(points) - len(verdicts))

    return {
        "score": score_from_coverage(verdicts),
        "coverage": verdicts,
        "strengths": [_to_student(s) for s in raw.strengths or [] if _clean(s)][:3],
        "improvements": [_to_student(s) for s in raw.improvements or [] if _clean(s)][:3],
        "model_answer": item.get("example_answer", ""),
    }
