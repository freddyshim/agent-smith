from typing import Optional, List
import tempfile
import os
from dotenv import load_dotenv
import asyncpg
import json
from datetime import datetime

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import PGVector
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import AsyncClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# Global variables
vectorstore = None
retriever = None
db_pool = None

# PostgreSQL connection string from environment
DATABASE_URL = os.getenv("DATABASE_URL")


# Embeddings
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


# API
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await init_database()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class DocumentUploadParams(BaseModel):
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    

class DocumentResponse(BaseModel):
    message: str
    num_documents: int
    num_chunks: int


class ChatRequest(BaseModel):
    prompt: str
    k: Optional[int] = 1
    session_id: str = "default"
    use_chat_history: Optional[bool] = True


# Database functions
async def init_database():
    """Initialize database connection pool and create tables"""
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    
    # Read and execute schema
    with open('db/schema.sql', 'r') as f:
        schema = f.read()
    
    async with db_pool.acquire() as conn:
        await conn.execute(schema)


async def store_chat_message(session_id: str, message_type: str, content: str, metadata: Optional[dict] = None):
    """Store a chat message in the database"""
    if db_pool is None:
        return
    
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chat_logs (session_id, message_type, content, metadata)
            VALUES ($1, $2, $3, $4)
            """,
            session_id, message_type, content, json.dumps(metadata or {})
        )


async def get_chat_history(session_id: str, limit: int = 10) -> List[dict]:
    """Retrieve recent chat history for a session"""
    if db_pool is None:
        return []
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT message_type, content, timestamp, metadata
            FROM chat_logs
            WHERE session_id = $1
            ORDER BY timestamp ASC
            LIMIT $2
            """,
            session_id, limit
        )
    
    return [{
        'message_type': row['message_type'],
        'content': row['content'],
        'timestamp': row['timestamp'],
        'metadata': json.loads(row['metadata'])
    } for row in rows]


# Methods
async def stream_chat_completion(msg: str):
    global llm
    async for part in await AsyncClient().chat(
        model="gemma3", messages=[{"role": "user", "content": msg}], stream=True
    ):
        token = part.message.content
        if isinstance(token, str):
            yield token


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Routes
@app.get("/")
def read_root():
    return "Hello, world!"


@app.get("/chat/{session_id}")
async def get_full_chat_history(session_id: str):
    """Retrieve full chat history for a session"""
    try:
        history = await get_chat_history(session_id, limit=1000)  # Get all messages
        return {"messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = 1000,
    chunk_overlap: Optional[int] = 200
):
    global vectorstore, retriever
    
    # Validate file type
    if file.filename is not None and not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Read the uploaded file content
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load PDF document
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            splits = text_splitter.split_documents(docs)
            
            # Create vectorstore
            vectorstore = PGVector.from_documents(
                documents=splits, 
                embedding=SentenceTransformerEmbeddings(),
                connection_string=DATABASE_URL,
                collection_name="document_embeddings"
            )
            retriever = vectorstore.as_retriever()
            
            return DocumentResponse(
                message="PDF document indexed successfully",
                num_documents=len(docs),
                num_chunks=len(splits)
            )
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def read_item(body: ChatRequest, response_model=StreamingResponse):
    global retriever, vectorstore
    
    if retriever is None or vectorstore is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Please index documents first.")
    
    try:
        # Store user message
        await store_chat_message(body.session_id, "user", body.prompt)
        
        # Update retriever with k parameter
        retriever = vectorstore.as_retriever(search_kwargs={"k": body.k})
        
        # Get context documents
        context_docs = retriever.get_relevant_documents(body.prompt)
        context = format_docs(context_docs)
        
        # Get chat history if requested
        chat_history = ""
        if body.use_chat_history:
            history = await get_chat_history(body.session_id, limit=6)  # Last 3 exchanges
            if history:
                chat_history = "\n\nPrevious conversation context:\n"
                for msg in history:
                    role = "User" if msg['message_type'] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n"
        
        # Create full prompt
        full_prompt = f"""You are an assistant that answers questions based on the context given. Use three sentences maximum and keep the answer concise. Do not refer to the provided context directly, but rather use the information to shape your answer. Do not answer questions that are irrelevant to the given context but instead say that you don't know the answer to the question.

Question: {body.prompt}

Context: {context}{chat_history}
"""
        
        # Get answer from Ollama and store it
        async def stream_and_store():
            response_parts = []
            async for token in stream_chat_completion(full_prompt):
                response_parts.append(token)
                yield token
            
            # Store complete response
            full_response = ''.join(response_parts)
            await store_chat_message(body.session_id, "assistant", full_response)
        
        return StreamingResponse(stream_and_store())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

