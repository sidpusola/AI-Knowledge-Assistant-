from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import shutil

from chains.tutor_chain import ask_tutor
from rag.loaders import load_document
from rag.splitter import split_documents
from rag.vectorstore import crea


app=FastAPI()

class QuestionRequest(BaseModel):
    question:str

@app.post("/ask")
def ask_question(request:QuestionRequest):
    answer=ask_tutor(request.question)

    return {
        "question":request.question,
        "answer":answer
    }    