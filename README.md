# PDF Chat Backend

FastAPI backend for PDF processing and AI-powered question answering using Google Gemini.

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## Setup

### 1. Environment Setup
```bash
# Copy environment file
cp .env.example .env

# Edit .env file and add your GEMINI_API_KEY
# GEMINI_API_KEY="your-api-key-here"
```

### 2. Quick Start
```bash
# Make start script executable
chmod +x start.sh

# Run the application
./start.sh
```

### 3. Manual Setup (Alternative)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Usage

1. Start the server using any method above
2. Open `http://localhost:8000/docs` for API documentation
3. Upload a PDF and ask questions

## API Endpoints

- `POST /upload-pdf` - Upload and process PDF file
- `POST /ask-question` - Ask questions about uploaded PDF

## Features

- PDF text extraction and processing
- AI-powered question answering
- CORS enabled for frontend integration
- Interactive API documentation

**API key issues:**
- Check your GEMINI_API_KEY in .env file
- Ensure no extra spaces or quotes