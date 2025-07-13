#!/usr/bin/env python3
"""Flask frontend for Voice Conversational AI"""

from flask import Flask, render_template, request, jsonify, send_file
import requests
import os
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Backend API configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

@app.route('/')
def index():
    """Main page with voice conversation interface"""
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Proxy transcribe request to backend"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Forward to backend
        files = {'audio': (audio_file.filename, audio_file.stream, audio_file.content_type)}
        response = requests.post(f'{BACKEND_URL}/transcribe', files=files)
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Proxy chat request to backend"""
    try:
        data = request.get_json()
        response = requests.post(f'{BACKEND_URL}/chat', json=data)
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak():
    """Proxy speak request to backend"""
    try:
        data = request.get_json()
        response = requests.post(f'{BACKEND_URL}/speak', json=data)
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Speak error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/converse', methods=['POST'])
def converse():
    """Proxy converse request to backend"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        conversation_id = request.form.get('conversation_id', 'default')
        
        # Forward to backend
        files = {'audio': (audio_file.filename, audio_file.stream, audio_file.content_type)}
        data = {'conversation_id': conversation_id}
        
        response = requests.post(f'{BACKEND_URL}/converse', files=files, data=data)
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Converse error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Proxy reset request to backend"""
    try:
        data = request.get_json()
        response = requests.post(f'{BACKEND_URL}/reset', json=data)
        
        return jsonify(response.json()), response.status_code
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from backend"""
    try:
        # Proxy audio file request to backend
        response = requests.get(f'{BACKEND_URL}/static/audio/{filename}')
        
        if response.status_code == 200:
            # Create temporary file and serve it
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                return send_file(tmp_file.name, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'Audio file not found'}), 404
            
    except Exception as e:
        logger.error(f"Audio serve error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)