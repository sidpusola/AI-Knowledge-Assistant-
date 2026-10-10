from contextlib import asynccontextmanager
from fastapi import Depends,FastAPI,UploadFile,HTTPException,File,Request,Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse,StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4
import json
import logging
import shutil

import httpx


def httpx_query(params:dict) -> str:
    """Query string with values escaped - an error message from
    Google can contain spaces and punctuation."""
    return str(httpx.QueryParams(params))

from langchain_core.messages import AIMessage, HumanMessage

from chains.tutor_chain import ask_tutor, stream_tutor
from rag.ingest import ingest_document
from llm.provider import LLMUnavailableError
from monitoring import get_stats
from feedback import FeedbackRequest, FeedbackSummary, feedback_summary, record_feedback
import auth
from database.store import (
    complete_profile, delete_session, init_db, list_sessions,
    load_transcript, user_stats,
)
from config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app:FastAPI):
    """Load the models at boot rather than on the first question.

    The embedding model and the cross-encoder take a few seconds each to
    build. Left lazy, that lands on whoever asks first - one student
    waiting ~15s while everyone after waits ~85ms. Paying it at startup
    moves the cost somewhere nobody is watching.
    """
    # uvicorn configures its own loggers but not the root one, so
    # anything this app logs - [auth], [db], [rewrite], [diagram] -
    # would otherwise go nowhere.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(message)s",
    )

    from rag.embeddings import get_embeddings
    from rag.vectorstore import load_vectorstore

    init_db()

    print("warming models...")
    get_embeddings().embed_query("warmup")
    load_vectorstore().similarity_search("warmup", k=1)

    if settings.USE_RERANKER:
        from rag.reranker import get_cross_encoder
        get_cross_encoder().predict([("warmup", "warmup")])

    print("models ready")
    yield


app=FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR=Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)              # creates your-project/uploads/

# Conversation memory, per session. In-memory only - resets on server
# restart and doesn't share across multiple worker processes. That's an
# intentional simplification (there's no database yet); a durable,
# multi-worker-safe version is real Week 7 scope (session persistence).
SESSIONS:dict[str,list[HumanMessage|AIMessage]]={}
MAX_HISTORY_MESSAGES=20  # keep memory bounded per session

class QuestionRequest(BaseModel):
    question:str
    session_id:str|None=None
    # Restrict retrieval to one subject, e.g. "life skills". None searches all.
    category:str|None=None

class UrlRequest(BaseModel):
    url:str

@app.get("/")
def home():
    return{
        "message":"AI knowledge assistant API is running"
    }


def _record_turn(session_id:str, history:list, question:str, answer:str) -> None:
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    if len(history) > MAX_HISTORY_MESSAGES:
        SESSIONS[session_id]=history[-MAX_HISTORY_MESSAGES:]


@app.post("/ask")
def ask_question(request:QuestionRequest):
    question=request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    session_id=request.session_id or str(uuid4())
    history=SESSIONS.setdefault(session_id, [])

    try:
        result=ask_tutor(question, history, request.category)
    except LLMUnavailableError as e:
        # Ollama isn't running, or the configured model isn't pulled -
        # a client-side/deployment problem, not a bug in the request.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        # Anything unexpected (bad retriever state, vectorstore error, etc.)
        # - don't leak an internal traceback to the client.
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer."
        )

    _record_turn(session_id, history, question, result["answer"])

    return {
        "question":question,
        "answer":result["answer"],
        "sources":result["sources"],
        "session_id":session_id
    }


@app.post("/ask/stream")
def ask_question_stream(request:QuestionRequest):
    question=request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    session_id=request.session_id or str(uuid4())
    history=SESSIONS.setdefault(session_id, [])

    def event_stream():
        # Server-Sent Events: each event is "event: <type>\ndata: <json>\n\n".
        # Once this generator starts yielding, the HTTP status is already
        # 200 and can't change - so unlike /ask, a failure here is reported
        # as an "error" event inside the stream, not an HTTP error status.
        chunks=[]
        sources=[]
        try:
            for chunk in stream_tutor(question, history, request.category, sources=sources):
                chunks.append(chunk)
                yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
        except LLMUnavailableError as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
            return
        except Exception:
            yield f"event: error\ndata: {json.dumps('Something went wrong while generating the answer.')}\n\n"
            return

        answer="".join(chunks)
        _record_turn(session_id, history, question, answer)

        yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'sources': sources})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}")
def reset_session(session_id:str):
    SESSIONS.pop(session_id, None)
    return {"message":"Conversation reset"}


class ProfileRequest(BaseModel):
    display_name:str
    # ISO "YYYY-MM-DD", or omitted. Validated in auth.parse_dob.
    date_of_birth:str|None=None


@app.get("/auth/status")
def auth_status():
    """Whether Google sign-in is set up on this server.

    The frontend asks this before showing a sign-in button, so a missing
    client ID reads as "not configured yet" rather than a broken button.
    """
    return {"configured":auth.is_configured()}


@app.get("/auth/google/login")
def google_login(request:Request):
    """Send the browser to Google's account chooser.

    First make sure we are on the host Google will return to. The CSRF
    state lives in a cookie, cookies are scoped by host, and a browser
    treats localhost and 127.0.0.1 as different hosts - so starting
    here on one and coming back on the other loses the cookie, and the
    callback can only say invalid_state. Bouncing to the canonical host
    first means that cannot happen, whichever address was typed.
    """
    try:
        expected=auth.callback_host()
    except auth.AuthNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if request.url.netloc != expected:
        return RedirectResponse(str(request.url.replace(netloc=expected)), status_code=307)

    try:
        url,state=auth.build_login_url()
    except auth.AuthNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    response=RedirectResponse(url, status_code=307)
    auth.set_state_cookie(response, state)
    return response


@app.get("/auth/google/callback")
async def google_callback(request:Request, code:str|None=None,
                          state:str|None=None, error:str|None=None):
    """Where Google sends the browser back.

    Always ends in a redirect to the app rather than a JSON body - a
    person is looking at this URL, not a script. Failures come back as
    ?auth_error= so the UI can say what went wrong.
    """
    def back(**params) -> RedirectResponse:
        target=settings.FRONTEND_URL
        if params:
            target=f"{target}?{httpx_query(params)}"
        response=RedirectResponse(target, status_code=303)
        auth.clear_state_cookie(response)
        return response

    if error:
        # The usual one is the student pressing Cancel.
        return back(auth_error=error)

    returned_cookie=request.cookies.get("aika_oauth_state")
    # Which of the two failure modes happened matters: a missing cookie
    # is a browser that didn't send it back (an embedded webview, or a
    # sign-in begun on another host); a present-but-rejected one is a
    # stale or forged value. They are fixed differently, so say which.
    logger.info(
        "[auth] callback: state=%s cookie=%s cookies_seen=%s",
        "present" if state else "missing",
        "present" if returned_cookie else "MISSING",
        list(request.cookies.keys()) or "none",
    )

    if not auth.verify_state(state, returned_cookie):
        # Nearly always a lost cookie rather than an attack: a sign-in
        # begun on a different host, or left open past the ten-minute
        # window. Both are fixed by starting again.
        return back(auth_error="Sign-in expired or was started elsewhere. Please try again.")

    if not code:
        return back(auth_error="missing_code")

    try:
        user=await auth.exchange_code_for_user(code)
    except auth.AuthNotConfigured:
        return back(auth_error="not_configured")
    except HTTPException as e:
        return back(auth_error=str(e.detail))

    response=back(signed_in="1")
    auth.issue_session(response, user.id)
    return response


@app.get("/auth/me")
def auth_me(request:Request):
    """The signed-in user, or null. Never 401s: the frontend calls this
    on load to decide which screen to show."""
    user=auth.current_user(request)
    return {"user":auth.public_user(user) if user else None}


@app.post("/auth/profile")
def save_profile(request:ProfileRequest, user=Depends(auth.require_user)):
    """The name and date of birth asked for after the first sign-in."""
    name=request.display_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Please enter a name.")

    updated=complete_profile(user.id, name, auth.parse_dob(request.date_of_birth))
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"user":auth.public_user(updated)}


@app.post("/auth/logout")
def logout(response:Response):
    auth.clear_session(response)
    return {"message":"Signed out"}


@app.get("/me/sessions")
def my_sessions(user=Depends(auth.require_user)):
    """This user's conversations, newest first."""
    return {"sessions":list_sessions(user.id)}


@app.get("/me/sessions/{session_id}")
def my_session(session_id:str, user=Depends(auth.require_user)):
    """One conversation in full, so it can be reopened."""
    messages=load_transcript(session_id, user.id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"session_id":session_id, "messages":messages}


@app.delete("/me/sessions/{session_id}")
def delete_my_session(session_id:str, user=Depends(auth.require_user)):
    if not delete_session(session_id, user.id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message":"Conversation deleted"}


@app.get("/me/stats")
def my_stats(user=Depends(auth.require_user)):
    """Activity for this student - sessions, questions, first and last
    seen. The per-user view the NGO needs for reporting."""
    return user_stats(user.id)


@app.post("/feedback")
def submit_feedback(request:FeedbackRequest):
    """Record a thumbs up or down on an answer.

    Fire-and-forget by design: it returns 200 whether or not the write
    succeeded, because a student rating an answer should never see an
    error for their trouble, and a lost rating costs nothing the system
    depends on. Failures are logged (see feedback.py).
    """
    record_feedback(request)
    return {"message":"Thanks for the feedback"}


@app.get("/feedback/summary", response_model=FeedbackSummary)
def feedback_stats():
    """Counts, satisfaction rate, and how ratings split between grounded
    and general-knowledge answers - the correlation worth watching."""
    return feedback_summary()


@app.get("/stats")
def stats():
    """Timing breakdown for recent requests - retrieval vs LLM vs total,
    plus time-to-first-token for streamed ones. See monitoring.py."""
    return get_stats()


@app.post("/ingest-url")
def ingest_url(request:UrlRequest):
    url=request.url.strip()

    if not url.lower().startswith(("http://","https://")):
        raise HTTPException(
            status_code=400,
            detail="URL must start with http:// or https://"
        )

    try:
        chunk_count=ingest_document(url)
    except Exception as e:
        # Fetching a page fails in many more ways than reading a local
        # file - DNS, timeouts, 404s, pages that are entirely JavaScript.
        # Reported back rather than leaking a traceback.
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch or process '{url}': {type(e).__name__}: {e}"
        )

    if chunk_count==0:
        return {
            "url":url,
            "message":"Page fetched, but no readable text was found "
                      "(it may render its content with JavaScript) - nothing was indexed."
        }

    return {
        "url":url,
        "chunks":chunk_count,
        "message":"Page fetched and indexed successfully"
    }


@app.get("/documents")
def list_documents():
    """What is actually searchable, not what happens to be on disk.

    This used to list files in uploads/, which was wrong in both
    directions: a page ingested from a URL never appeared because it
    isn't a file, and clearing the index still showed every PDF as
    though students could ask about it.
    """
    from rag.service import service

    return {"documents": service.list_documents()}


@app.delete("/documents/{doc_id}")
def delete_document_endpoint(doc_id:str):
    """Remove a document from the knowledge base.

    Deletes the chunks, which is what makes it unsearchable. The original
    file in uploads/ is left alone - deleting the user's upload is a
    bigger decision than un-indexing it, and is not what this asks for.
    """
    from rag.service import service

    document=service.get_document(doc_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed document with id '{doc_id}'"
        )

    removed=service.delete(doc_id)

    return {
        "doc_id":doc_id,
        "filename":document["filename"],
        "chunks_removed":removed,
        "message":f"Removed '{document['filename']}' from the knowledge base"
    }


@app.post("/upload")
async def upload_document(file: UploadFile=File(...)):
    allowed_extensions={
        ".txt",
        ".md",
        ".pdf",
        ".docx",
        ".csv"
    }
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File must have a filename"
        )
    suffix=Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type :{suffix}"
        )

    file_path= UPLOAD_DIR/file.filename

    try:
        with open(file_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save uploaded file: {e}"
        )

    try:
        chunk_count=ingest_document(str(file_path))
    except Exception as e:
        # Loader/parsing failures vary a lot by file type (corrupt PDF,
        # malformed CSV, unreadable DOCX, ...) - too many exception types
        # across 4 different loaders to enumerate individually, so this
        # boundary catches broadly and reports back what went wrong
        # instead of letting a raw traceback reach the client.
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Could not process '{file.filename}': {e}"
        )

    if chunk_count==0:
        return {
            "filename":file.filename,
            "message":"File uploaded, but no readable text was found in it "
                       "(e.g. a scanned/image-only PDF) - nothing was indexed."
        }

    return {
        "filename":file.filename,
        "message":"Documents uploaded and indexed successfully"
    }
