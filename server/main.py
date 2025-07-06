from typing import Optional, List
import tempfile
import os

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import AsyncClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# Global variables
vectorstore = None
retriever = None


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
            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=SentenceTransformerEmbeddings()
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
def read_item(body: ChatRequest, response_model=StreamingResponse):
    global retriever, vectorstore
    
    if retriever is None or vectorstore is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Please index documents first.")
    
    try:
        # Update retriever with k parameter
        retriever = vectorstore.as_retriever(search_kwargs={"k": body.k})
        
        # Get context documents
        context_docs = retriever.get_relevant_documents(body.prompt)
        context = format_docs(context_docs)
        
        # Create full prompt
        full_prompt = f"""You are an assistant that answers questions based on the context given. Use three sentences maximum and keep the answer concise. Do not refer to the provided context directly, but rather use the information to shape your answer. Do not answer questions that are irrelevant to the given context but instead say that you don't know the answer to the question.

Question: {body.prompt}

Context: {context}
"""
        
        # Get answer from Ollama
        return StreamingResponse(stream_chat_completion(full_prompt))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

