# API Contracts Documentation

## Voice Conversational AI REST API

### Base URL
```
http://localhost:8000
```

---

## Endpoints

### 1. Health Check
**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### 2. Transcribe Audio
**POST** `/transcribe`

**Request:**
- Content-Type: `multipart/form-data`
- Body: Audio file (WAV, MP3, M4A, FLAC, OGG)

**Example:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "text": "Hello, how are you?",
  "transcription_time": 1.23,
  "filename": "audio_1234567890.wav"
}
```

---

### 3. Chat with LLM + RAG
**POST** `/chat`

**Request:**
```json
{
  "conversation_id": "user123_session1",
  "message": "Tell me about 36 W 36th St",
  "context": "optional additional context"
}
```

**Response:**
```json
{
  "response": "36 W 36th St is managed by John Doe. The rent is $5,000/month...",
  "conversation_id": "user123_session1",
  "chat_time": 2.45,
  "rag_context": "Found 3 relevant documents about this property...",
  "resolved_references": {
    "that property": "36 W 36th St"
  }
}
```

---

### 4. Text-to-Speech
**POST** `/speak`

**Request:**
```json
{
  "text": "Hello, how can I help you today?",
  "voice": "alloy"
}
```

**Available voices:** `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

**Response:**
```json
{
  "audio_url": "/static/audio/tts_1234567890_alloy.mp3",
  "tts_time": 0.89,
  "filename": "tts_1234567890_alloy.mp3"
}
```

---

### 5. End-to-End Conversation
**POST** `/converse`

**Request:**
```json
{
  "conversation_id": "user123_session1",
  "message": "What properties are available in Times Square?",
  "voice": "alloy"
}
```

**Response:**
```json
{
  "text_response": "I found several properties in Times Square...",
  "audio_url": "/static/audio/tts_1234567890_alloy.mp3",
  "conversation_id": "user123_session1",
  "chat_time": 2.15,
  "tts_time": 1.02,
  "total_time": 3.17,
  "rag_context": "Found 5 relevant documents...",
  "resolved_references": {}
}
```

---

### 6. Reset Conversation
**POST** `/reset`

**Request:**
```json
{
  "conversation_id": "user123_session1"
}
```

**Response:**
```json
{
  "message": "Conversation reset successfully",
  "conversation_id": "user123_session1"
}
```

---

### 7. Upload RAG Documents
**POST** `/upload_rag_docs`

**Request:**
- Content-Type: `multipart/form-data`
- Body: Document files (PDF, TXT, CSV, JSON)

**Example:**
```bash
curl -X POST "http://localhost:8000/upload_rag_docs" \
  -F "files=@document1.pdf" \
  -F "files=@document2.csv"
```

**Response:**
```json
{
  "message": "2 documents uploaded and indexed successfully",
  "processed_files": ["document1.pdf", "document2.csv"],
  "extraction_time": 1.45,
  "processing_time": 3.21,
  "indexing_time": 2.67,
  "total_time": 7.33
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message description"
}
```

**Common HTTP Status Codes:**
- `400` - Bad Request (invalid input)
- `422` - Unprocessable Entity (validation error)
- `500` - Internal Server Error

---

## Sample Usage Scenarios

### Scenario 1: Simple Voice Query
```bash
# 1. Upload audio
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@question.wav"

# 2. Get response with RAG
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test1", "message": "Who manages 36 W 36th St?"}'

# 3. Convert to speech
curl -X POST "http://localhost:8000/speak" \
  -H "Content-Type: application/json" \
  -d '{"text": "John Doe manages that property", "voice": "alloy"}'
```

### Scenario 2: End-to-End Conversation
```bash
curl -X POST "http://localhost:8000/converse" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "voice1", 
    "message": "What is the rent for 9 Times Square?",
    "voice": "nova"
  }'
```

### Scenario 3: Upload and Query Documents
```bash
# 1. Upload new documents
curl -X POST "http://localhost:8000/upload_rag_docs" \
  -F "files=@new_properties.csv"

# 2. Query the uploaded data
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "doc_test", "message": "What new properties were just added?"}'
```

---

## Performance Metrics

All endpoints include timing information:

- **transcription_time**: Time to convert speech to text
- **chat_time**: Time for LLM processing + RAG retrieval  
- **tts_time**: Time to generate speech audio
- **total_time**: End-to-end processing time
- **extraction_time**: Time to extract text from documents
- **processing_time**: Time to process and chunk documents
- **indexing_time**: Time to create vector embeddings and store

---

## Rate Limits

- Maximum file size: 25MB per upload
- Maximum audio length: 10 minutes
- Request rate: 100 requests per minute per IP

---

## Authentication

Currently no authentication required for development.
Production deployment should implement API key authentication.