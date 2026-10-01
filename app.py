from fastapi import FastAPI,UploadFile,HTTPException,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4
import json
import shutil

from langchain_core.messages import AIMessage, HumanMessage

from chains.tutor_chain import ask_tutor, stream_tutor
from rag.ingest import ingest_document
from llm.provider import LLMUnavailableError
from monitoring import get_stats
from config import settings


app=FastAPI()

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
        answer=ask_tutor(question, history)
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

    _record_turn(session_id, history, question, answer)

    return {
        "question":question,
        "answer":answer,
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
        try:
            for chunk in stream_tutor(question, history):
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

        yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}")
def reset_session(session_id:str):
    SESSIONS.pop(session_id, None)
    return {"message":"Conversation reset"}


@app.get("/stats")
def stats():
    """Timing breakdown for recent requests - retrieval vs LLM vs total,
    plus time-to-first-token for streamed ones. See monitoring.py."""
    return get_stats()


@app.get("/documents")
def list_documents():
    documents=sorted(p.name for p in UPLOAD_DIR.iterdir() if p.is_file())
    return {"documents":documents}


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
