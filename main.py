from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from src.clients import api_clients

app = FastAPI(
    title="Voice Conversational AI API",
    description="RESTful API for voice conversations with LLM enhanced by RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# explicitly handle /favicon.ico so browsers find it
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/")
async def root():
    return {"message": "Voice Conversational AI API is running"}

@app.get("/health")
async def health_check():
    try:
        openai_status = api_clients.test_openai_connection()
        qdrant_status = api_clients.test_qdrant_connection()
        
        if openai_status and qdrant_status:
            return {
                "status": "healthy", 
                "message": "API is operational",
                "services": {
                    "openai": "connected",
                    "qdrant": "connected"
                }
            }
        else:
            return {
                "status": "degraded",
                "message": "Some services unavailable",
                "services": {
                    "openai": "connected" if openai_status else "disconnected",
                    "qdrant": "connected" if qdrant_status else "disconnected"
                }
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Service check failed: {str(e)}",
            "services": {
                "openai": "unknown",
                "qdrant": "unknown"
            }
        }

@app.post("/transcribe")
async def transcribe_audio():
    return {"message": "Transcribe endpoint - TODO"}

@app.post("/chat")
async def chat():
    return {"message": "Chat endpoint - TODO"}

@app.post("/speak")
async def text_to_speech():
    return {"message": "Speak endpoint - TODO"}

@app.post("/converse")
async def converse():
    return {"message": "Converse endpoint - TODO"}

@app.post("/reset")
async def reset_conversation():
    return {"message": "Reset endpoint - TODO"}

@app.post("/upload_rag_docs")
async def upload_documents():
    return {"message": "Upload RAG docs endpoint - TODO"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)