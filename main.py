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
from typing import Optional
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
app.mount("/static/audio", StaticFiles(directory="audio_output"), name="audio")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class TranscribeResponse(BaseModel):
    text: str
    transcription_time: float
    filename: str

class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    processing_time: float
    rag_context_used: bool
    rag_sources: Optional[list] = None

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"  # Default voice
    speed: float = 1.0    # Speech speed (0.25 to 4.0)

class TTSResponse(BaseModel):
    audio_file_path: str
    audio_url: str
    generation_time: float
    text_length: int
    voice_used: str
    file_size_bytes: int

class ConverseResponse(BaseModel):
    transcribed_text: str
    chat_response: str
    audio_file_path: str
    audio_url: str
    conversation_id: str
    timing: dict  # Contains breakdown of each stage
    rag_context_used: bool
    rag_sources: Optional[list] = None

class ResetRequest(BaseModel):
    conversation_id: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float = 0.3

class TopKRequest(BaseModel):
    query: str
    k: int = 3

class MetadataSearchRequest(BaseModel):
    query: str
    source_contains: str = None
    document_type: str = None
    last_n_days: int = None
    limit: int = 5

# Initialize services
doc_processor = DocumentProcessor()
rag_service = RAGService()

# In-memory conversation storage
conversations: dict = {}

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

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio file to text using OpenAI Whisper API"""
    start_time = time.time()
    
    try:
        # Validate file type
        file_ext = Path(audio.filename).suffix.lower()
        supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.webm', '.mp4'}
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(supported_formats)}"
            )
        
        # Validate file size (OpenAI Whisper has 25MB limit)
        audio_content = await audio.read()
        if len(audio_content) > 25 * 1024 * 1024:  # 25MB
            raise HTTPException(
                status_code=400,
                detail="Audio file too large. Maximum size is 25MB."
            )
        
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe using OpenAI Whisper
            transcription_start = time.time()
            
            with open(temp_file_path, "rb") as audio_file:
                transcript = api_clients.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            transcription_time = time.time() - transcription_start
            total_time = time.time() - start_time
            
            logger.info(f"Transcribed {audio.filename} in {transcription_time:.2f}s")
            
            return TranscribeResponse(
                text=transcript,
                transcription_time=transcription_time,
                filename=audio.filename
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with LLM enhanced by RAG system"""
    start_time = time.time()
    logger.info(f"🚀 Chat request started - conversation_id: {request.conversation_id}, message: '{request.message[:50]}...'")
    
    try:
        # Get or create conversation history first (needed for RAG reference resolution)
        logger.info("💾 Managing conversation history...")
        if request.conversation_id not in conversations:
            conversations[request.conversation_id] = []
            logger.info(f"🆕 Created new conversation: {request.conversation_id}")
        
        conversation_history = conversations[request.conversation_id]
        logger.info(f"📜 Conversation history length: {len(conversation_history)} messages")

        # Retrieve relevant context from RAG system if no context provided
        rag_context_used = False
        rag_sources = []
        context_text = request.context
        
        logger.info(f"📝 Context provided: {bool(request.context)}")
        
        if not context_text:
            try:
                logger.info("🔍 Starting RAG retrieval with reference resolution...")
                # Get relevant context from RAG system
                rag_start = time.time()
                
                # Get relevant context from RAG system with conversation history for reference resolution
                try:
                    rag_results = await rag_service.get_contextualized_response_data(
                        query=request.message,
                        k=3,
                        conversation_history=conversation_history  # Pass conversation history for reference resolution
                    )
                    rag_time = time.time() - rag_start
                    logger.info(f"✅ RAG retrieval completed in {rag_time:.2f}s")
                except Exception as rag_error:
                    rag_time = time.time() - rag_start
                    logger.error(f"❌ RAG retrieval failed after {rag_time:.2f}s: {rag_error}")
                    rag_results = None
                
                if rag_results and rag_results.get("formatted_context"):
                    context_text = rag_results["formatted_context"]
                    # Deduplicate sources
                    all_sources = [doc.get("source", "unknown") for doc in rag_results.get("retrieved_documents", [])]
                    rag_sources = list(set(all_sources))  # Remove duplicates
                    rag_context_used = True
                    logger.info(f"📚 RAG context retrieved: {len(context_text)} chars, sources: {rag_sources}")
                else:
                    if rag_results is None:
                        logger.warning("⚠️ RAG retrieval failed, proceeding without context")
                    else:
                        logger.warning(f"⚠️ No RAG context found in results: {rag_results.keys() if rag_results else 'None'}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to retrieve RAG context: {e}")
        
        # Build messages for OpenAI
        logger.info("🔧 Building OpenAI messages...")
        messages = []
        
        # System prompt
        system_prompt = """You are a helpful real estate assistant AI. You have access to a comprehensive database of property listings, broker information, and real estate market data.

When users ask about properties, provide detailed, accurate information based on the context provided. If you don't have specific information, say so clearly.

IMPORTANT: When users ask about a specific address, ONLY use information for that EXACT address. Do not confuse similar addresses (e.g., "36 W 36th St" vs "7 W 36th St" are completely different properties).

Be conversational, helpful, and professional. Focus on:
- Property details (price, location, features, etc.)
- Market insights and trends  
- Broker and agent information
- Helpful real estate advice

Always be truthful about what information you have access to and ensure address accuracy."""

        if context_text:
            system_prompt += f"\n\nRelevant context from property database:\n{context_text}"
            logger.info(f"🔗 Added RAG context to system prompt ({len(context_text)} chars)")
            logger.info(f"📄 Context preview: {context_text[:500]}...")
        
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for msg in conversation_history:
            messages.append(msg)
        logger.info(f"📚 Added {len(conversation_history)} history messages")
        
        # Add current user message
        user_message = {"role": "user", "content": request.message}
        messages.append(user_message)
        logger.info(f"💬 Total messages for OpenAI: {len(messages)}")
        
        # Call OpenAI GPT-4o Mini
        logger.info("🤖 Starting OpenAI API call...")
        llm_start = time.time()
        try:
            response = api_clients.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                timeout=30  # Add 30 second timeout
            )
            llm_time = time.time() - llm_start
            logger.info(f"✅ OpenAI API call completed in {llm_time:.2f}s")
            
            assistant_response = response.choices[0].message.content
            logger.info(f"📝 Response generated: {len(assistant_response)} characters")
            
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed after {time.time() - llm_start:.2f}s: {e}")
            raise
        
        # Update conversation history
        logger.info("💾 Updating conversation history...")
        conversation_history.append(user_message)
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Keep conversation history manageable (last 10 messages)
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
            conversations[request.conversation_id] = conversation_history
            logger.info("🧹 Trimmed conversation history to last 10 messages")
        
        total_time = time.time() - start_time
        
        logger.info(f"🎉 Chat response generated in {total_time:.2f}s (LLM: {llm_time:.2f}s)")
        
        return ChatResponse(
            response=assistant_response,
            conversation_id=request.conversation_id,
            processing_time=total_time,
            rag_context_used=rag_context_used,
            rag_sources=rag_sources if rag_sources else None
        )
        
    except Exception as e:
        total_error_time = time.time() - start_time
        logger.error(f"💥 Chat endpoint failed after {total_error_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/speak", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using OpenAI TTS API"""
    start_time = time.time()
    
    try:
        # Validate voice selection
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if request.voice not in valid_voices:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice '{request.voice}'. Valid options: {', '.join(valid_voices)}"
            )
        
        # Validate speed
        if not (0.25 <= request.speed <= 4.0):
            raise HTTPException(
                status_code=400,
                detail="Speed must be between 0.25 and 4.0"
            )
        
        # Validate text length (OpenAI TTS has a 4096 character limit)
        if len(request.text) > 4096:
            raise HTTPException(
                status_code=400,
                detail="Text too long. Maximum 4096 characters allowed."
            )
        
        if len(request.text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        # Create audio_output directory if it doesn't exist
        audio_dir = Path("audio_output")
        audio_dir.mkdir(exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        filename = f"tts_{timestamp}_{request.voice}.mp3"
        file_path = audio_dir / filename
        
        logger.info(f"🔊 Generating TTS: voice={request.voice}, speed={request.speed}, text_length={len(request.text)}")
        
        # Generate speech using OpenAI TTS
        tts_start = time.time()
        
        response = api_clients.openai_client.audio.speech.create(
            model="tts-1",
            voice=request.voice,
            input=request.text,
            speed=request.speed
        )
        
        # Save audio to file
        with open(file_path, "wb") as audio_file:
            for chunk in response.iter_bytes():
                audio_file.write(chunk)
        
        tts_time = time.time() - tts_start
        total_time = time.time() - start_time
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Create URL for accessing the audio file
        audio_url = f"/static/audio/{filename}"
        
        logger.info(f"✅ TTS generated in {tts_time:.2f}s, file size: {file_size} bytes")
        
        return TTSResponse(
            audio_file_path=str(file_path),
            audio_url=audio_url,
            generation_time=tts_time,
            text_length=len(request.text),
            voice_used=request.voice,
            file_size_bytes=file_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/converse", response_model=ConverseResponse)
async def converse(
    conversation_id: str = Form(...),
    audio: UploadFile = File(...),
    voice: str = Form("alloy"),
    speed: float = Form(1.0)
):
    """End-to-end voice conversation: audio → transcribe → chat → speak → audio"""
    total_start = time.time()
    timing = {}
    
    try:
        logger.info(f"🎙️ Starting voice conversation for {conversation_id}")
        
        # Stage 1: Transcribe audio to text
        transcribe_start = time.time()
        
        # Validate audio file
        file_ext = Path(audio.filename).suffix.lower()
        supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.webm', '.mp4'}
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {file_ext}. Supported: {', '.join(supported_formats)}"
            )
        
        # Read and validate audio content
        audio_content = await audio.read()
        if len(audio_content) > 25 * 1024 * 1024:  # 25MB limit
            raise HTTPException(
                status_code=400,
                detail="Audio file too large. Maximum size is 25MB."
            )
        
        # Transcribe audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as audio_file:
                transcript = api_clients.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            transcribed_text = transcript.strip()
            timing["transcribe_time"] = time.time() - transcribe_start
            logger.info(f"🎯 Transcribed: '{transcribed_text}' in {timing['transcribe_time']:.2f}s")
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Stage 2: Chat with LLM + RAG
        chat_start = time.time()
        
        # Create chat request
        chat_request = ChatRequest(
            conversation_id=conversation_id,
            message=transcribed_text
        )
        
        # Get chat response (reuse existing chat logic)
        chat_response_obj = await chat(chat_request)
        chat_response_text = chat_response_obj.response
        rag_context_used = chat_response_obj.rag_context_used
        rag_sources = chat_response_obj.rag_sources
        
        timing["chat_time"] = time.time() - chat_start
        logger.info(f"💬 Chat response generated in {timing['chat_time']:.2f}s")
        
        # Stage 3: Convert response to speech
        tts_start = time.time()
        
        # Validate voice and speed
        valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        if voice not in valid_voices:
            voice = "alloy"  # Default fallback
        
        if not (0.25 <= speed <= 4.0):
            speed = 1.0  # Default fallback
        
        # Create TTS request
        tts_request = TTSRequest(
            text=chat_response_text,
            voice=voice,
            speed=speed
        )
        
        # Generate speech (reuse existing TTS logic)
        tts_response_obj = await text_to_speech(tts_request)
        
        timing["tts_time"] = time.time() - tts_start
        timing["total_time"] = time.time() - total_start
        
        logger.info(f"🔊 TTS generated in {timing['tts_time']:.2f}s")
        logger.info(f"🎉 Full conversation pipeline completed in {timing['total_time']:.2f}s")
        
        return ConverseResponse(
            transcribed_text=transcribed_text,
            chat_response=chat_response_text,
            audio_file_path=tts_response_obj.audio_file_path,
            audio_url=tts_response_obj.audio_url,
            conversation_id=conversation_id,
            timing=timing,
            rag_context_used=rag_context_used,
            rag_sources=rag_sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - total_start
        logger.error(f"💥 Conversation pipeline failed after {total_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")

@app.post("/reset")
async def reset_conversation(request: ResetRequest):
    """Reset conversation memory for a specific conversation ID"""
    try:
        conversation_id = request.conversation_id
        
        if conversation_id in conversations:
            message_count = len(conversations[conversation_id])
            del conversations[conversation_id]
            logger.info(f"🧹 Reset conversation {conversation_id} ({message_count} messages cleared)")
            
            return {
                "message": f"Conversation {conversation_id} reset successfully",
                "conversation_id": conversation_id,
                "messages_cleared": message_count
            }
        else:
            return {
                "message": f"Conversation {conversation_id} not found (already empty)",
                "conversation_id": conversation_id,
                "messages_cleared": 0
            }
            
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

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

@app.post("/clear_and_reindex")
async def clear_and_reindex():
    """Clear Qdrant collection and re-index the default CSV file"""
    try:
        start_time = time.time()
        
        # Delete existing collection
        logger.info("🗑️ Deleting existing Qdrant collection...")
        delete_success = rag_service.delete_collection()
        if not delete_success:
            raise Exception("Failed to delete collection")
        
        # Recreate collection
        logger.info("🏗️ Creating new collection...")
        create_success = rag_service.ensure_collection_exists()
        if not create_success:
            raise Exception("Failed to create new collection")
        
        # Re-index the CSV file
        csv_file = "rag_data/HackathonInternalKnowledgeBase.csv"
        if os.path.exists(csv_file):
            logger.info(f"📁 Re-indexing {csv_file}...")
            
            # Extract documents from CSV
            documents = doc_processor.extract_text(csv_file, csv_file)
            
            # Process into chunks
            processed_docs = doc_processor.process_documents(documents)
            
            # Index in vector database
            result = rag_service.index_documents(processed_docs)
            
            total_time = time.time() - start_time
            
            return {
                "message": "Collection cleared and re-indexed successfully",
                "documents_processed": len(documents),
                "chunks_indexed": result["indexed"],
                "collection": result["collection"],
                "total_time": total_time
            }
        else:
            raise Exception(f"CSV file not found: {csv_file}")
            
    except Exception as e:
        logger.error(f"Error clearing and re-indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Clear and re-index failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)