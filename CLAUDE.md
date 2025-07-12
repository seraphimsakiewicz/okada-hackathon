# Voice Conversational Agentic AI

**Project Requirements & Submission Guidelines**

---

## 🎯 Objective

Design and implement a Python-based RESTful API application that enables bi-directional **voice conversations** with a Large Language Model (LLM), enhanced with **Retrieval Augmented Generation (RAG)** using proprietary/internal documents.

---

## ✅ Core Requirements

### 1. Voice Input & Transcription

* Capture real-time voice input from the microphone
* Convert speech to text:
  * Use OpenAI Whisper API
* Return **transcription time** in seconds

### 2. LLM Text Processing

* Use OpenAI GPT-4o Mini with good structured output.
* Maintain **conversation memory** (retain previous context)
* Support **parameter injection** of RAG context with the message

### 3. RAG Agent Integration

* Accept proprietary/internal documents through a **dedicated upload endpoint**
* Index documents using vector store:
  * Qdrant Cloud
* Retrieve relevant documents dynamically so TTS can use info from uploaded docs

### 4. Text-to-Speech (TTS)

* Convert LLM output text into audio:
  * Using OpenAI TTS
  * Play generated speech through speaker
  * Report **TTS generation time**

### 5. RESTful API Design

| Endpoint           | Method | Description                                                    |
| ------------------ | ------ | -------------------------------------------------------------- |
| `/transcribe`      | POST   | Accepts audio, returns text + STT time                         |
| `/chat`            | POST   | Accepts conversation + new message + context, returns response |
| `/speak`           | POST   | Accepts text, returns audio + TTS time                         |
| `/converse`        | POST   | End-to-end pipeline (voice → LLM + RAG → audio)                |
| `/reset`           | POST   | Clears conversation memory                                     |
| `/upload_rag_docs` | POST   | Uploads RAG knowledge base documents (PDF, TXT, CSV, JSON)     |

---

## 📦 Submission Checklist

* RESTful API app in Python (FastAPI/Flask preferred)
* GitHub repo link (add collaborators, usernames will be shared)
* Functional voice-to-text → LLM (with RAG) → text-to-speech pipeline
* Working document upload and indexing via `/upload_rag_docs`
* All endpoints must return **processing durations**

> **Note:** You are always welcome to go beyond the requirements once the core is complete.

---

## 📁 Add to GitHub Repo

* `README.md` with setup and usage instructions
* `.env` file with **placeholder** values (no real API keys)
* Architecture diagram (Mermaid or similar)
* Sample conversation logs (if applicable)

## Deployment
* Use Fly.io

---

## 📄 API Contracts (.pdf file)

* Input formats
* Response schema
* Sample API calls and usage notes


