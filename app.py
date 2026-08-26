from fastapi import FastAPI,UploadFile,HTTPException,File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil

from chains.tutor_chain import ask_tutor
from rag.ingest import ingest_document


app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR=Path("uploads")                  
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
    answer=ask_tutor(request.question)

    return {
        "question":request.question,
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

    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    ingest_document(str(file_path))  

    return {
        "filename":file.filename,
        "message":"Documents uploaded and indexed successfully"
    }                

