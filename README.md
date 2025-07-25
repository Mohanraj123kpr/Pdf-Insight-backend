# PDF Chat Backend

FastAPI backend for PDF processing and AI-powered question answering using Google Gemini.

## Setup

1. **Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Server**
   ```bash
   # Option 1: Using the startup script
   ./start.sh
   
   # Option 2: Direct command
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `POST /upload-pdf` - Upload and process PDF file
- `POST /ask-question` - Ask questions about the uploaded PDF
- `GET /health` - Health check

## Features

- PDF text extraction using PyPDF2
- Text chunking for better processing
- FAISS vector search for relevant context
- Google Gemini AI for question answering
- CORS enabled for frontend integration