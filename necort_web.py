from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the Nash Equilibrium version of RecThink
from nash_recursive_thinking import NashEquilibriumRecursiveChat

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NECoRT API", description="Nash-Equilibrium Chain of Recursive Thoughts")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a dictionary to store chat instances
chat_instances = {}

# Pydantic models for request/response validation
class ChatConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
    num_agents: int = 3
    convergence_threshold: float = 0.05

class MessageRequest(BaseModel):
    session_id: str
    message: str
    thinking_rounds: Optional[int] = None

class SaveRequest(BaseModel):
    session_id: str
    filename: Optional[str] = None
    full_log: bool = False

@app.post("/api/initialize")
async def initialize_chat(config: ChatConfig):
    """Initialize a new NECoRT session"""
    try:
        # Generate a session ID
        session_id = f"necort_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
        
        # Use API key from environment if not provided in the request
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No API key provided. Please set OPENROUTER_API_KEY in .env file or provide it in the request.")
        
        # Initialize the Nash Equilibrium chat instance
        chat = NashEquilibriumRecursiveChat(
            api_key=api_key, 
            model=config.model,
            num_agents=config.num_agents,
            convergence_threshold=config.convergence_threshold
        )
        chat_instances[session_id] = chat
        
        return {
            "session_id": session_id, 
            "status": "initialized",
            "system_type": "NECoRT - Nash Equilibrium Chain of Recursive Thoughts"
        }
    except Exception as e:
        logger.error(f"Error initializing NECoRT: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize NECoRT: {str(e)}")

@app.post("/api/send_message")
async def send_message(request: MessageRequest):
    """Send a message and get a response with Nash Equilibrium thinking process"""
    try:
        if request.session_id not in chat_instances:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chat = chat_instances[request.session_id]
        
        # Override thinking rounds if provided
        original_thinking_fn = chat._determine_thinking_rounds
        
        if request.thinking_rounds is not None:
            # Override the thinking rounds determination
            chat._determine_thinking_rounds = lambda _: request.thinking_rounds
        
        # Process the message
        result = chat.think_and_respond(request.message, verbose=True)
        
        # Restore original function
        chat._determine_thinking_rounds = original_thinking_fn
        
        return {
            "session_id": request.session_id,
            "response": result["response"],
            "thinking_rounds": result["thinking_rounds"],
            "thinking_history": result["thinking_history"],
            "converged": result.get("converged", False),
            "convergence_round": result.get("convergence_round"),
            "final_response_agent": result.get("final_response_agent")
        }
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

@app.post("/api/save")
async def save_conversation(request: SaveRequest):
    """Save the conversation or full thinking log"""
    try:
        if request.session_id not in chat_instances:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chat = chat_instances[request.session_id]
        
        filename = request.filename
        if request.full_log:
            chat.save_nash_equilibrium_log(filename)
        else:
            chat.save_conversation(filename)
        
        return {"status": "saved", "filename": filename}
    except Exception as e:
        logger.error(f"Error saving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save conversation: {str(e)}")

@app.get("/api/sessions")
async def list_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, chat in chat_instances.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(chat.conversation_history) // 2,  # Each message-response pair counts as 2
            "created_at": session_id.split("_")[1],  # Extract timestamp from session ID
            "is_nash_equilibrium": isinstance(chat, NashEquilibriumRecursiveChat)
        })
    
    return {"sessions": sessions}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in chat_instances:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_instances[session_id]
    return {"status": "deleted", "session_id": session_id}

# WebSocket for streaming thinking process
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in chat_instances:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    chat = chat_instances[session_id]
    
    try:
        # Set up a custom callback to stream thinking process
        original_call_api = chat._call_api
        
        async def stream_callback(chunk):
            await websocket.send_json({"type": "chunk", "content": chunk})
        
        # Override the _call_api method to also send updates via WebSocket
        def ws_call_api(messages, temperature=0.7, stream=True):
            result = original_call_api(messages, temperature, stream)
            # Send the chunk via WebSocket if we're streaming
            if stream:
                asyncio.create_task(stream_callback(result))
            return result
        
        # Replace the method temporarily
        chat._call_api = ws_call_api
        
        # Wait for messages from the client
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "message":
                # Process the message
                result = chat.think_and_respond(message_data["content"], verbose=True)
                
                # Send the final result
                await websocket.send_json({
                    "type": "final",
                    "response": result["response"],
                    "thinking_rounds": result["thinking_rounds"],
                    "thinking_history": result["thinking_history"],
                    "converged": result.get("converged", False),
                    "convergence_round": result.get("convergence_round", None),
                    "final_response_agent": result.get("final_response_agent", 0)
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Restore original function
        if chat and hasattr(chat, '_call_api'):
            chat._call_api = original_call_api

@app.get("/")
async def root():
    return {
        "name": "NECoRT API",
        "description": "Nash-Equilibrium Chain of Recursive Thoughts",
        "version": "1.0.0",
        "tagline": "Let your thoughts argue, evolve, and stabilize."
    }

if __name__ == "__main__":
    # Create the .env file if it doesn't exist
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Add your OpenRouter API key here\n")
            f.write("OPENROUTER_API_KEY=\n")
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  NECoRT - Nash-Equilibrium Chain of Recursive Thoughts    ║
    ║  "Let your thoughts argue, evolve, and stabilize."        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("Starting NECoRT API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 