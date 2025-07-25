from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import PyPDF2
import io
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="PDF Chat API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
pdf_chunks = []
faiss_index = None

class QuestionRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    matched_chunks: List[str]

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embedding(text: str):
    """Get embedding for text"""
    return embed_model.encode([text])[0]

def build_faiss_index(chunks: List[str]):
    """Build FAISS index from text chunks"""
    global faiss_index
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings).astype("float32"))

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    global pdf_chunks
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read PDF content
        content = await file.read()
        pdf_file = io.BytesIO(content)
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        
        # Create chunks
        pdf_chunks = chunk_text(text)
        
        # Build FAISS index
        build_faiss_index(pdf_chunks)
        
        return {
            "message": "PDF processed successfully",
            "chunks_count": len(pdf_chunks),
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask-question", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Ask question about the uploaded PDF"""
    global pdf_chunks, faiss_index
    
    logger.info(f"Received question: {request.question}")
    
    if not pdf_chunks or faiss_index is None:
        logger.error("No PDF uploaded yet")
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")
    
    try:
        # Get query embedding
        logger.info("Getting query embedding...")
        query_vec = get_embedding(request.question)
        
        # Search similar chunks
        logger.info("Searching similar chunks...")
        D, I = faiss_index.search(np.array([query_vec]).astype("float32"), k=3)
        
        # Get matched chunks
        matched_chunks = [pdf_chunks[i] for i in I[0]]
        logger.info(f"Found {len(matched_chunks)} matched chunks")
        
        # Build context
        context = "\n".join(matched_chunks)
        
        # Create prompt
        prompt = f"""You are an intelligent assistant.
Use the context below to answer the question accurately.

Context:
{context}

Question:
{request.question}

Answer:"""
        
        # Get response from Gemini
        logger.info("Calling Gemini API...")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini API")
        
        logger.info("Successfully generated response")
        return ChatResponse(
            answer=response.text.strip(),
            matched_chunks=matched_chunks
        )
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)