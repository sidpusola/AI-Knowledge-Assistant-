from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app:FastAPI):
    """Load the models at boot rather than on the first question.

    The embedding model and the cross-encoder take a few seconds each to
    build. Left lazy, that lands on whoever asks first - one student
    waiting ~15s while everyone after waits ~85ms. Paying it at startup
    moves the cost somewhere nobody is watching.
    """
    from rag.embeddings import get_embeddings
    from rag.vectorstore import load_vectorstore

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
