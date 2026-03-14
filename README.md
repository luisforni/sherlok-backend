# voice-conscience backend

Minimal FastAPI backend that accepts audio uploads and provides a WebSocket endpoint for streaming responses.

Quick start (local):

1. Create a virtualenv and install deps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run locally:
   ```bash
   uvicorn main:app --reload
   ```

Notes:
- `main.py` contains placeholder hooks for Whisper (transcription) and LLM streaming — replace with your models.
