from fastapi import FastAPI,UploadFile,HTTPException,File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil

from chains.tutor_chain import ask_tutor
from rag.ingest import ingest_document
from llm.provider import LLMUnavailableError
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

class QuestionRequest(BaseModel):
    question:str

@app.get("/")
def home():
    return{
        "message":"AI knowledge assistant API is running"
    }


@app.post("/ask")
def ask_question(request:QuestionRequest):
    question=request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        answer=ask_tutor(question)
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

    return {
        "question":question,
        "answer":answer
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
