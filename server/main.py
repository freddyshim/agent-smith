from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ollama import AsyncClient, ChatResponse, chat
from pydantic import BaseModel

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


# Schema
class ChatBody(BaseModel):
    prompt: str


# Methods
async def stream_chat_completion(msg: str):
    async for part in await AsyncClient().chat(
        model="gemma3", messages=[{"role": "user", "content": msg}], stream=True
    ):
        token = part.message.content
        if isinstance(token, str):
            yield token


# Routes
@app.get("/")
def read_root():
    return "Hello, world!"


@app.post("/chat")
def read_item(body: ChatBody):
    return StreamingResponse(stream_chat_completion(body.prompt))
