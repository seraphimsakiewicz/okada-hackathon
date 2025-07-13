from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import os
import tempfile
import time
from pathlib import Path
from src.clients import api_clients
from src.services.document_processor import DocumentProcessor
from src.services.rag_service import RAGService

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
doc_processor = DocumentProcessor()
rag_service = RAGService()

@app.on_event("startup")
async def startup_event():
    """Initialize the application and index default documents"""
    logger.info("Starting Voice Conversational AI API...")
    
    # Check if HackathonInternalKnowledgeBase.csv exists and index it
    csv_file = "rag_data/HackathonInternalKnowledgeBase.csv"
    if os.path.exists(csv_file):
        try:
            logger.info(f"{csv_file} found, refraining from reindexing for now...")
            # logger.info(f"Found {csv_file}, indexing automatically...")
            # start_time = time.time()
            
            # # Extract documents from CSV
            # documents = doc_processor.extract_text(csv_file, csv_file)
            
            # # Process into chunks
            # processed_docs = doc_processor.process_documents(documents)
            
            # # Index in vector database
            # result = rag_service.index_documents(processed_docs)
            
            # end_time = time.time()
            # logger.info(f"Indexed {result['indexed']} chunks from {csv_file} in {end_time - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to index {csv_file}: {e}")
    else:
        logger.warning(f"{csv_file} not found, skipping automatic indexing")
    
    logger.info("API startup complete")

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
async def upload_documents(file: UploadFile = File(...)):
    """Upload and index documents for RAG system"""
    start_time = time.time()
    
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        supported_formats = {'.pdf', '.txt', '.csv', '.json'}
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(supported_formats)}"
            )
        
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract text from the file
            extraction_start = time.time()
            documents = doc_processor.extract_text(temp_file_path, file.filename)
            extraction_time = time.time() - extraction_start
            
            # Process documents into chunks
            processing_start = time.time()
            processed_docs = doc_processor.process_documents(documents)
            processing_time = time.time() - processing_start
            
            # Index documents in vector database
            indexing_start = time.time()
            indexing_result = rag_service.index_documents(processed_docs)
            indexing_time = time.time() - indexing_start
            
            total_time = time.time() - start_time
            
            return {
                "message": "Document uploaded and indexed successfully",
                "filename": file.filename,
                "file_size": len(content),
                "documents_extracted": len(documents),
                "chunks_created": len(processed_docs),
                "indexed_count": indexing_result["indexed"],
                "collection": indexing_result["collection"],
                "timing": {
                    "extraction_time": extraction_time,
                    "processing_time": processing_time,
                    "indexing_time": indexing_time,
                    "total_time": total_time
                }
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float = 0.3

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search for relevant documents using RAG system"""
    try:
        start_time = time.time()
        results = rag_service.search_documents(
            query=request.query, 
            limit=request.limit, 
            score_threshold=request.score_threshold
        )
        search_time = time.time() - start_time
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "search_time": search_time
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/collection_info")
async def get_collection_info():
    """Get information about the current RAG collection"""
    try:
        info = rag_service.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

class TopKRequest(BaseModel):
    query: str
    k: int = 3

@app.post("/search/top_k")
async def get_top_relevant_chunks(request: TopKRequest):
    """Get top-k most relevant document chunks optimized for LLM context"""
    try:
        start_time = time.time()
        results = rag_service.get_top_relevant_chunks(
            query=request.query, 
            k=request.k
        )
        search_time = time.time() - start_time
        
        return {
            "query": request.query,
            "k": request.k,
            "results": results,
            "count": len(results),
            "search_time": search_time
        }
        
    except Exception as e:
        logger.error(f"Error getting top-k results: {e}")
        raise HTTPException(status_code=500, detail=f"Top-k search failed: {str(e)}")

@app.post("/search/context")
async def get_contextualized_data(request: TopKRequest):
    """Get retrieved documents and formatted context for LLM injection"""
    try:
        start_time = time.time()
        result_data = rag_service.get_contextualized_response_data(
            query=request.query, 
            k=request.k
        )
        processing_time = time.time() - start_time
        
        return {
            **result_data,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error getting contextualized data: {e}")
        raise HTTPException(status_code=500, detail=f"Context generation failed: {str(e)}")

class MetadataSearchRequest(BaseModel):
    query: str
    source_contains: str = None
    document_type: str = None
    last_n_days: int = None
    limit: int = 5

@app.post("/search/metadata")
async def search_by_metadata(request: MetadataSearchRequest):
    """Search documents with metadata filtering capabilities"""
    try:
        start_time = time.time()
        results = rag_service.search_by_metadata(
            query=request.query,
            source_contains=request.source_contains,
            document_type=request.document_type,
            last_n_days=request.last_n_days,
            limit=request.limit
        )
        search_time = time.time() - start_time
        
        return {
            "query": request.query,
            "filters": {
                "source_contains": request.source_contains,
                "document_type": request.document_type,
                "last_n_days": request.last_n_days
            },
            "results": results,
            "count": len(results),
            "search_time": search_time
        }
        
    except Exception as e:
        logger.error(f"Error searching by metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Metadata search failed: {str(e)}")

@app.get("/sources")
async def get_document_sources():
    """Get list of all unique document sources in the collection"""
    try:
        sources = rag_service.get_document_sources()
        return {
            "sources": sources,
            "count": len(sources)
        }
    except Exception as e:
        logger.error(f"Error getting document sources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sources: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)