### Phase 1: Core Infrastructure (2-3 hours)
1. **Setup FastAPI application structure** ✅
   
   **Implementation Details:**
   - [x] Create main.py with FastAPI app instance and basic route handlers
   - [x] Setup requirements.txt with all dependencies (fastapi, uvicorn, openai, qdrant-client, python-multipart, pydantic, python-dotenv)
   - [x] Create .env template with placeholder API keys (OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY)
   - [x] Create project folders: /src, /uploads, /audio_output
   
   **Manual Testing:**
   - [x] Run `pip install -r requirements.txt` to verify dependencies install correctly
   - [x] Run `uvicorn main:app --reload` to start server
   - [x] Visit http://localhost:8000/docs to see auto-generated API documentation
   - [x] Test basic health check endpoint: `curl http://localhost:8000/health`

2. **Configure API integrations** ✅
   
   **Implementation Details:**
   - [x] Create OpenAI client instance with API key from environment variables
   - [x] Setup Qdrant client connection with cloud URL and API key
   - [x] Create configuration management module to handle all environment variables
   - [x] Add error handling for missing API keys and connection failures
   
   **Manual Testing:**
   - [x] Add real API keys to .env file
   - [x] Create test script that checks OpenAI API connection: `python test_openai.py`
   - [x] Create test script that verifies Qdrant connection: `python test_qdrant.py`

### Phase 2: RAG System Implementation (2-3 hours)
3. **Document processing and indexing**
   
   **Implementation Details:**
   - [x] Create `/upload_rag_docs` endpoint that accepts multipart file uploads (PDF, TXT, CSV, JSON)
   - [x] Implement document text extraction (pypdf2 for PDF, pandas for CSV, json for JSON)
   - [x] Create text chunking function (overlap sliding window, 500-1000 chars per chunk)
   - [x] Generate embeddings using OpenAI text-embedding-ada-002 model
   - [x] Setup Qdrant collection with vector dimensions matching OpenAI embeddings (1536)
   - [x] Index the existing HackathonInternalKnowledgeBase.csv automatically on startup
   
   **Manual Testing:**
   - [x] Test file upload: `curl -X POST -F "file=@test.pdf" http://localhost:8000/upload_rag_docs`
   - [x] Test CSV upload: `curl -X POST -F "file=@HackathonInternalKnowledgeBase.csv" http://localhost:8000/upload_rag_docs`
   - [x] Check Qdrant dashboard/console to verify vectors are indexed
   - [x] Test with sample text file: create test.txt and upload via Swagger UI at /docs

4. **RAG retrieval system**
   
   **Implementation Details:**
   - [x] Create semantic search function that queries Qdrant with embedding similarity
   - [x] Implement relevance scoring and top-k retrieval (top 3-5 most relevant chunks)
   - [x] Create context formatting function that prepares retrieved docs for LLM injection
   - [x] Add metadata filtering capabilities (document source, upload date)
   
   **Manual Testing:**
   - [x] Create test endpoint `/search` that takes a query and returns relevant documents
   - [x] Added additional test endpoints: `/search/top_k`, `/search/context`, `/search/metadata`, `/sources`
   - [x] Test search: `curl -X POST -H "Content-Type: application/json" -d '{"query": "real estate pricing"}' http://localhost:8000/search`
   - [x] Verify search returns relevant property data from the CSV
   - [x] Test different query types: location-based, price-based, broker-based searches
   - [x] Fixed Qdrant filter validation issues and implemented post-filtering fallback
   - [x] Verified all 455 documents are properly indexed and searchable

### Phase 3: Core API Endpoints (3-4 hours)
5. **Speech-to-Text endpoint**
   
   **Implementation Details:**
   - [ ] Create `/transcribe` endpoint that accepts audio file uploads (WAV, MP3, M4A)
   - [ ] Integrate OpenAI Whisper API for speech-to-text conversion
   - [ ] Add timing measurement (start/end timestamps)
   - [ ] Handle audio file validation and temporary storage
   - [ ] Return JSON response with transcribed text and processing duration
   
   **Manual Testing:**
   - [ ] Record audio file using phone/computer (say "Hello, what properties are available?")
   - [ ] Test upload: `curl -X POST -F "audio=@test_audio.wav" http://localhost:8000/transcribe`
   - [ ] Use Swagger UI at /docs to upload audio file through browser
   - [ ] Verify response includes both transcription text and timing in seconds
   - [ ] Test with different audio formats and lengths

6. **Chat functionality**
   
   **Implementation Details:**
   - [ ] Create `/chat` endpoint accepting conversation_id, message, and optional context
   - [ ] Implement conversation memory using in-memory dictionary (conversation_id -> message history)
   - [ ] Integrate RAG system to retrieve relevant context for each message
   - [ ] Setup OpenAI GPT-4o Mini API calls with system prompts and conversation history
   - [ ] Format response with RAG context injection and conversation flow
   
   **Manual Testing:**
   - [ ] Test basic chat: `curl -X POST -H "Content-Type: application/json" -d '{"conversation_id": "test1", "message": "Hello"}' http://localhost:8000/chat`
   - [ ] Test with real estate query: `curl -X POST -H "Content-Type: application/json" -d '{"conversation_id": "test1", "message": "What properties are available for under $500k?"}' http://localhost:8000/chat`
   - [ ] Verify conversation memory by sending follow-up questions in same conversation_id
   - [ ] Check that responses include relevant property information from uploaded CSV

7. **Text-to-Speech endpoint**
   
   **Implementation Details:**
   - [ ] Create `/speak` endpoint that accepts text input
   - [ ] Integrate OpenAI TTS API with voice selection (alloy, echo, fable, onyx, nova, shimmer)
   - [ ] Generate audio file and save to /audio_output directory
   - [ ] Add timing measurement for TTS generation
   - [ ] Return audio file path/URL and processing duration
   
   **Manual Testing:**
   - [ ] Test TTS: `curl -X POST -H "Content-Type: application/json" -d '{"text": "Hello, I found 3 properties matching your criteria"}' http://localhost:8000/speak`
   - [ ] Download and play the generated audio file
   - [ ] Test with longer text responses from the chat endpoint
   - [ ] Verify timing metrics are accurate and reasonable

8. **End-to-end pipeline**
   
   **Implementation Details:**
   - [ ] Create `/converse` endpoint combining transcribe → chat → speak pipeline
   - [ ] Accept audio upload, process through all three stages
   - [ ] Maintain conversation state throughout the pipeline
   - [ ] Return final audio response with timing breakdown for each stage
   - [ ] Implement `/reset` endpoint to clear conversation memory for specific conversation_id
   
   **Manual Testing:**
   - [ ] Record question: "What properties are available in downtown area?"
   - [ ] Test full pipeline: `curl -X POST -F "audio=@question.wav" -F "conversation_id=test1" http://localhost:8000/converse`
   - [ ] Verify you receive audio response with property information
   - [ ] Test conversation continuity with follow-up audio questions
   - [ ] Test reset: `curl -X POST -H "Content-Type: application/json" -d '{"conversation_id": "test1"}' http://localhost:8000/reset`

### Phase 4: Testing & Documentation (1-2 hours)
9. **API testing and validation**
   
   **Implementation Details:**
   - [ ] Create comprehensive test suite covering all endpoints
   - [ ] Test error handling for invalid inputs and missing API keys
   - [ ] Validate response schemas match API contracts
   - [ ] Performance testing for timing metrics accuracy
   - [ ] End-to-end integration testing of complete voice pipeline
   
   **Manual Testing:**
   - [ ] Run complete test script: `python test_all_endpoints.py`
   - [ ] Test error cases: invalid audio formats, missing conversation_id, large file uploads
   - [ ] Performance test: measure actual vs reported timing metrics
   - [ ] Load test: concurrent requests to /chat endpoint
   - [ ] Validate audio quality: play generated TTS files and verify clarity

10. **Documentation completion**
    
    **Implementation Details:**
    - [ ] Create comprehensive README.md with installation, setup, and usage instructions
    - [ ] Generate API documentation using FastAPI's automatic OpenAPI schema
    - [ ] Create architecture diagram showing data flow between components
    - [ ] Document environment variables and configuration options
    - [ ] Add troubleshooting section with common issues and solutions
    
    **Manual Testing:**
    - [ ] Follow README.md setup instructions on a fresh environment
    - [ ] Verify all curl commands in documentation work correctly
    - [ ] Test API documentation at http://localhost:8000/docs
    - [ ] Validate architecture diagram matches actual implementation
    - [ ] Ensure all required environment variables are documented

### Phase 5: Deployment (1 hour)
11. **Fly.io deployment**
    
    **Implementation Details:**
    - [ ] Install Fly.io CLI and authenticate account
    - [ ] Create Dockerfile for containerized deployment
    - [ ] Generate fly.toml configuration file with app settings
    - [ ] Configure environment variables in Fly.io secrets
    - [ ] Setup persistent volume for audio file storage
    - [ ] Deploy application with proper health checks
    
    **Manual Testing:**
    - [ ] Install flyctl: `curl -L https://fly.io/install.sh | sh`
    - [ ] Authenticate: `fly auth login`
    - [ ] Initialize app: `fly launch` (follow prompts)
    - [ ] Set secrets: `fly secrets set OPENAI_API_KEY=your_key QDRANT_URL=your_url QDRANT_API_KEY=your_key`
    - [ ] Deploy: `fly deploy`
    - [ ] Test production endpoints: `curl https://your-app.fly.dev/health`
    - [ ] Test full voice pipeline on production URL
    - [ ] Monitor logs: `fly logs` to verify deployment success
