import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import redis
import json
import uuid
import asyncio
from typing import List

from utils import get_vectorstore
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.runnables import RunnableLambda
from memory import RedisChatMessageHistory

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

system_prompt = """
You are a knowledgeable news assistant with access to a database of news articles.
- Always provide accurate, concise, and helpful responses based on the retrieved documents.
- When answering, cite specific information from the sources provided.
- Avoid repeating the same information in different ways
- If the answer is not in the provided documents, acknowledge this and provide general information if possible.
- Maintain context awareness across the conversation and refer back to previous topics when relevant.
- Format your responses clearly, using bullet points for lists and keeping paragraphs concise.

Retrieved information is below:
{context}

Previous conversation:
{chat_history}
"""

question_prompt = """

Based on the conversation history and retrieved documents, please answer the following question:
{question}

Remember to:
1. Stay focused on the question
2. Use information from the retrieved documents
3. Acknowledge when information might be incomplete
4. Reference your sources
"""

# Create prompt templates
qa_prompt = PromptTemplate(
    template=system_prompt + "\n" + question_prompt,
    input_variables=["context", "question", "chat_history"]
)

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    convert_system_message_to_human=True,
    model_kwargs={"stream_chat": True}
)

# Create the conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=get_vectorstore(),
    return_source_documents=True,
    return_generated_question=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    verbose=False
)

def format_response(response):
    """Process response with improved source deduplication"""
    import re
    from urllib.parse import urlparse

    # Initial cleanup
    answer = response['answer']
    answer = re.sub(r'^🤖:\s*', '', answer).strip()
    answer = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', answer)

    # Check if answer starts with "I am sorry"
    is_apology = answer.lower().startswith("i am sorry")

    # Heading formatting
    answer = re.sub(r'([^:\n]+):', r'**\1:**', answer)
    answer = re.sub(r'\*\*\*\*', r'**', answer)
    
    # Cleanup whitespace and formatting
    answer = re.sub(r'(\n\s*){2,}', '\n\n', answer)
    answer = re.sub(r'(•|\-)\s+', r'\n• ', answer)
    answer = re.sub(r'```(.*?)```', r'\n```\1\n```', answer, flags=re.DOTALL)

    # Content deduplication
    raw_parts = re.split(r'(?:\n\s*[\*\-•])|\n{2,}', answer)
    seen_content = set()
    sections = []
    for part in raw_parts:
        text = part.strip()
        if not text:
            continue
        key = re.sub(r'\W+', '', text.lower())
        if key in seen_content:
            continue
        seen_content.add(key)
        sections.append(text)
    answer = '• ' + '\n• '.join(sections)

    # Source processing with enhanced deduplication - skip if answer is an apology
    sources_section = ""
    if not is_apology and "source_documents" in response:
        sources_section = "\n\n📚 **Sources**\n\n"
        source_list = []
        seen_sources = {}  # Use dict instead of set to track by both title and URL
        
        for doc in response["source_documents"]:
            # Extract metadata
            title = doc.metadata.get("title", "").strip()
            raw_url = doc.metadata.get("url", "").strip()
            date = doc.metadata.get("date", "")[:10] if doc.metadata.get("date") else ""

            # Normalize URL and title for deduplication
            if raw_url:
                parsed_url = urlparse(raw_url)
                clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}".lower()
            else:
                clean_url = ""
                
            # Create unique content fingerprint from first 100 chars of doc content
            content_fingerprint = doc.page_content[:100].lower() if hasattr(doc, 'page_content') else ""
            
            # Deduplication using multiple keys
            source_key = (clean_url or "") + "|" + (title.lower() or "") + "|" + content_fingerprint
            
            # Skip if we've seen this source already
            if source_key in seen_sources:
                continue
                
            # Skip empty sources
            if not (clean_url or title):
                continue
                
            seen_sources[source_key] = True

            # Format source entry
            source_text = "• "
            if clean_url:
                display_text = title or parsed_url.netloc
                source_text += f"[{display_text}]({raw_url})\n\n"
            else:
                source_text += f"**{title or 'Untitled Source'}**\n\n"

            # Add metadata
            meta = []
            if date:
                meta.append(f"\n🗓️ {date}")
            
            if meta:
                source_text += f" ({' | '.join(meta)})"
            
            source_list.append(source_text)

        # Only add sources section if we have sources
        if source_list:
            sources_section += "\n\n".join(source_list)
        else:
            sources_section = ""

    # Final cleanup
    answer = re.sub(r'\s+([.,!?])', r'\1', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    
    # Add original query if available
    if "generated_question" in response:
        query = ' '.join(dict.fromkeys(response["generated_question"].split()))
        query = re.sub(r'\b(\w+)\s+\1\b', r'\1', query)

    return f"🤖: {answer}{sources_section}"
def format_sources(response):
    """Format sources separately from streaming response"""
    seen = set()
    sources = []
    for doc in response["source_documents"]:
        source = doc.metadata.get("url") or doc.metadata.get("title", "Unknown")
        if source not in seen:
            seen.add(source)
            meta = []
            if date := doc.metadata.get("date"):
                meta.append(f"🗓️ {date}")
            sources.append(f"- {source}" + (f" ({' | '.join(meta)})" if meta else ""))
    return "\n".join(sources)

streaming_chain = (
    RunnableLambda(lambda x: {"question": x["question"], "chat_history": x["chat_history"]})
    | qa_chain
    | RunnableLambda(format_response)
)

class SessionValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        session_id = request.path_params.get("session_id")
        if session_id and session_id != "new":
            if not redis_client.exists(f"chat:{session_id}"):
                raise HTTPException(status_code=404, detail="Session not found")
        
        return await call_next(request)

app = FastAPI()
app.add_middleware(SessionValidationMiddleware)
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class Message(BaseModel):
    role: str
    content: str
    timestamp: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ChatMessage(BaseModel):
    message: str

class ChatSession(BaseModel):
    session_id: str
    messages: List[Message]
    model_config = ConfigDict(arbitrary_types_allowed=True)

def get_memory_for_session(session_id: str):
    """Get memory for a specific session"""
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
        output_key="answer",
        chat_memory=RedisChatMessageHistory(session_id=session_id)
    )

def get_chat_history(session_id: str) -> List[Message]:
    """Retrieve chat history from Redis"""
    history = redis_client.get(f"chat:{session_id}")
    if history:
        return [Message(**msg) for msg in json.loads(history)]
    return []

def save_chat_history(session_id: str, messages: List[Message]):
    redis_client.set(
        f"chat:{session_id}",
        json.dumps([msg.model_dump() for msg in messages]),
        ex=86400  # Reset TTL on every update
    )
async def generate_streaming_response(formatted_response: str):
    """Generate chunks with improved markdown awareness and smoother streaming"""
    buffer = ""
    markdown_elements = ['*', '_', '`', '#', '>', '[', ']', '|', '-', '+']
    in_markdown = False
    
    for char in formatted_response:
        buffer += char
        
        # Keep markdown syntax elements together for better formatting
        if char in markdown_elements:
            in_markdown = not in_markdown
            continue
        
        # Handle end of sentence or list item - good breakpoints
        if not in_markdown and (
            (len(buffer) >= 5 and char in ['.', '!', '?', ';', ':']) or
            (len(buffer) >= 3 and buffer.endswith('\n'))
        ):
            yield {"event": "message", "data": buffer, "id": str(uuid.uuid4())}
            buffer = ""
            await asyncio.sleep(0.05)  # Slightly longer pause at sentence endings
        
        # Regular text chunking - break after a moderate buffer size
        elif not in_markdown and len(buffer) >= 40:
            # Try to break at word boundaries
            last_space = buffer.rfind(' ', 30)
            if last_space > 0:
                yield {"event": "message", "data": buffer[:last_space+1], "id": str(uuid.uuid4())}
                buffer = buffer[last_space+1:]
            else:
                yield {"event": "message", "data": buffer, "id": str(uuid.uuid4())}
                buffer = ""
            await asyncio.sleep(0.03)
    
    # Send any remaining text
    if buffer:
        yield {"event": "message", "data": buffer, "id": str(uuid.uuid4())}

@app.post("/chat/new")
async def create_chat_session():
    session_id = str(uuid.uuid4())
    # Initialize with empty array to prevent 404
    redis_client.set(
        f"chat:{session_id}",
        json.dumps([]),  # Initialize empty messages array
        ex=86400
    )
    return {"session_id": session_id}

@app.get("/chat/validate/{session_id}")
async def validate_session(session_id: str):
    exists = redis_client.exists(f"chat:{session_id}")
    return {"valid": bool(exists)}

@app.post("/chat/{session_id}")
async def chat_endpoint(session_id: str, chat_message: ChatMessage):
    try:
        # Get memory for this session
        memory = get_memory_for_session(session_id)
        
        # Add user message
        memory.chat_memory.add_user_message(chat_message.message)
        
        # Get response
        result = qa_chain({
            "question": chat_message.message,
            "chat_history": memory.chat_memory.messages[-10:]  # Get last exchanges
        })
        
        # Add assistant message
        memory.chat_memory.add_ai_message(result['answer'])
        
        # Format response
        formatted_response = format_response(result)
        return {"response": formatted_response}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class StreamChatMessage(BaseModel):
    chat_message: str


@app.post("/chat/{session_id}/stream")
async def stream_chat(session_id: str, message: StreamChatMessage):
    memory = get_memory_for_session(session_id)
    memory.chat_memory.add_user_message(message.chat_message)
    
    
    async def generate():
        full_response = ""
        try:
            # Get formatted response first
            formatted_response = format_response(await qa_chain.acall({
                "question": message.chat_message,
                "chat_history": memory.chat_memory.messages[-10:]
            }))

            # Stream formatted response preserving markdown
            buffer = ""
            in_code_block = False
            in_bold = False
            
            # Process character by character for better markdown preservation
            for char in formatted_response:
                buffer += char
                
                # Handle markdown code blocks
                if buffer.endswith('```'):
                    in_code_block = not in_code_block
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.02)
                    continue
                
                if buffer.endswith('**'):
                    in_bold = not in_bold
                    if not in_bold: 
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.02)
                    continue
                    
                if not in_code_block and not in_bold:
                    if char in ['\n', '.', '!', '?'] and len(buffer) > 3:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.03)  
                    elif char == ' ' and len(buffer) > 20:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.02)

                elif (in_code_block or in_bold) and len(buffer) > 50:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.02)

            if buffer:
                yield buffer

            memory.chat_memory.add_ai_message(formatted_response)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield "⚠️ An error occurred while generating the response"
            
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/chat/sessions")
async def get_all_sessions():
    """Retrieve all active session IDs from Redis"""
    session_keys = redis_client.keys("chat:*")
    
    session_ids = [key.split(":", 1)[1] for key in session_keys]
    
    return {"session_ids": session_ids, "count": len(session_ids)}

@app.get("/chat/{session_id}/history")
async def get_history(session_id: str):
    if not redis_client.exists(f"chat:{session_id}"):
        return JSONResponse(
            status_code=200,
            content={"messages": []}
        )
    messages = get_chat_history(session_id)
    return {"messages": messages}

@app.post("/chat/{session_id}/clear")
async def clear_history(session_id: str):
    redis_client.delete(f"chat:{session_id}")
    return {"message": "Chat history cleared"}

if __name__ == "__main__":
    import uvicorn
    from create_chroma import create_vector_db
    create_vector_db()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers = 1)