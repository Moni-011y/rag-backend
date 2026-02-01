import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from processor import process_file
from rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title="RAG Chat API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

print(f"Backend started. Storage in: {UPLOAD_DIR}")
rag_engine = RAGEngine(index_dir=INDEX_DIR)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    print(f"Received {len(files)} files for upload.")
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        print(f"Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            print(f"Processing file: {file.filename}")
            chunks = process_file(file_path)
            print(f"Adding {len(chunks)} chunks to vector store.")
            rag_engine.add_documents(chunks, file.filename)
            results.append({"filename": file.filename, "status": "success"})
            print(f"Successfully indexed: {file.filename}")
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
            results.append({"filename": file.filename, "status": "error", "message": str(e)})
            
    return {"results": results}

@app.get("/documents")
async def list_documents():
    return {"documents": rag_engine.get_indexed_files()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"Received chat request: {request.message} (Session: {request.session_id})")
    try:
        response, sources = rag_engine.query(request.message, request.session_id)
        print("Successfully generated response.")
        return ChatResponse(response=response, sources=sources)
    except Exception as e:
        print(f"Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    # In a real app, you'd rebuild the index. For simplicity, we just remove from tracked list.
    success = rag_engine.remove_document(filename)
    if success:
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Document not found")
