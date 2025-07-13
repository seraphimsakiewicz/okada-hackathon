# Voice Conversational Agentic AI

A Python-based RESTful API application that enables bi-directional voice conversations with a Large Language Model (LLM), enhanced with Retrieval Augmented Generation (RAG) using proprietary real estate documents.

## 🎯 Features

- **Voice Input & Transcription**: Real-time voice capture with OpenAI Whisper API
- **LLM Text Processing**: OpenAI GPT-4o Mini with conversation memory
- **RAG Agent Integration**: Qdrant Cloud vector store for document retrieval
- **Text-to-Speech**: OpenAI TTS with multiple voice options
- **Smart Follow-up Queries**: Resolves references like "that property" to specific addresses
- **Granular Property Search**: Handles floor/suite specificity for multi-unit buildings

## 🏗️ Architecture

```
Voice Input → Whisper API → LLM + RAG → OpenAI TTS → Audio Output
                    ↑           ↓
               Conversation   Qdrant
                Memory       Vector DB
```

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transcribe` | POST | Voice to text conversion |
| `/chat` | POST | Text chat with LLM + RAG |
| `/speak` | POST | Text to speech conversion |
| `/converse` | POST | End-to-end voice conversation |
| `/reset` | POST | Clear conversation memory |
| `/upload_rag_docs` | POST | Upload knowledge base documents |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Qdrant Cloud account

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd okada-hackathon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the backend API:
```bash
python main.py
```

5. Run the Flask frontend (in a new terminal):
```bash
python frontend.py
```

The API will be available at `http://localhost:8000`
The Web UI will be available at `http://localhost:5000`

## 🔧 Configuration

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=voice_ai_documents
```

## 📚 Usage Examples

### Web Interface (Recommended)
Visit `http://localhost:5000` for the interactive web interface featuring:
- **Voice Recording**: Click to start/stop recording
- **Text Chat**: Type questions and get responses
- **Audio Playback**: Hear responses spoken back to you
- **Example Queries**: Click pre-made examples to test

### API Usage

#### Basic Chat
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"conversation_id": "test1", "message": "Hello"}' \
  http://localhost:8000/chat
```

#### Property Query
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"conversation_id": "test1", "message": "Who manages 36 W 36th St?"}' \
  http://localhost:8000/chat
```

#### Follow-up Query
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"conversation_id": "test1", "message": "What is the rent for that property?"}' \
  http://localhost:8000/chat
```

#### Voice Conversation
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"conversation_id": "voice1", "message": "Tell me about Times Square properties"}' \
  http://localhost:8000/converse
```

## 🧪 Testing

The project includes comprehensive testing tools:

### Test All Properties
```bash
# Test first 25 properties
python test_all_properties.py --limit 25

# Test specific range
python test_all_properties.py --start 199 --limit 25

# Test all properties
python test_all_properties.py
```

### Test Results
Recent comprehensive testing shows:
- **96% success rate** across diverse address formats
- **98% average accuracy** in associate matching
- **2.57s average response time**

## 🏢 RAG Knowledge Base

The system includes a real estate knowledge base with:
- **226 properties** across NYC
- **Multiple address formats**: Streets, Avenues, Broadway, Times Square
- **Granular details**: Floor and suite-level information
- **Associate information**: Property managers and brokers
- **Financial data**: Rent rates, annual/monthly costs

### Supported Address Formats
- Standard streets: "36 W 36th St"
- Avenues: "345 Seventh Avenue"
- Special locations: "9 Times Sq"
- Broadway addresses: "1412 Broadway"
- Complex ranges: "121-127 W 27th St"

## 🎤 Voice Features

### Supported Audio Formats
- WAV, MP3, M4A, FLAC, OGG

### TTS Voices
- alloy (default)
- echo
- fable
- onyx
- nova
- shimmer

## 🔍 Smart Follow-up Resolution

The system intelligently resolves follow-up queries:

**Example Conversation:**
1. User: "Who manages 9 Times Sq, Suite 3A, Floor P3?"
2. System: "Joshamee Gibbs, Sansa Stark, Sheldon Cooper, Sergio Perez"
3. User: "What is the rent for that property?"
4. System: Resolves "that property" → "9 Times Sq, Suite 3A, Floor P3"

## 📊 Performance Metrics

All endpoints return timing information:
- `transcribe_time`: Speech-to-text processing
- `chat_time`: LLM + RAG processing  
- `tts_time`: Text-to-speech generation
- `total_time`: End-to-end processing

## 🔒 Security

- API keys stored in environment variables
- No sensitive data in repository
- Rate limiting for API protection
- Input validation and error handling

## 📁 Project Structure

```
okada-hackathon/
├── main.py                 # FastAPI application
├── src/
│   ├── clients/           # API client configurations
│   ├── services/          # Core business logic
│   └── config.py          # Settings management
├── rag_data/              # Knowledge base documents
├── audio_output/          # Generated TTS files
├── test_all_properties.py # Comprehensive testing
└── requirements.txt       # Dependencies
```

## 🚀 Deployment

The application is configured for Fly.io deployment:

```bash
# Deploy to Fly.io
fly deploy
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Support

For questions or issues, please check the documentation or contact the development team.

---

🤖 **Generated with Claude Code** - Comprehensive voice AI system for real estate property management.
