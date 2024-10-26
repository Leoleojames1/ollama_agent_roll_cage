from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from typing import Optional, List
import ollama
from ollama_chatbot_base import ollama_chatbot_base
import asyncio

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot
chatbot = ollama_chatbot_base()

class ChatRequest(BaseModel):
    user_input: str
    model_name: str
    image: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

class ModelRequest(BaseModel):
    model_name: str

class CommandRequest(BaseModel):
    command: str

class ToggleRequest(BaseModel):
    feature: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            response = chatbot.process_message(data)
            await manager.send_personal_message(response, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client disconnected")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = chatbot.send_prompt(request.user_input)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swap_model")
async def swap_model(request: ModelRequest):
    try:
        chatbot.user_input_model_select = request.model_name
        chatbot.swap()
        return {"message": f"Model changed to {request.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/toggle_feature")
async def toggle_feature(request: ToggleRequest):
    try:
        if request.feature == "llava":
            chatbot.llava_flag = not chatbot.llava_flag
            return {"llava_status": chatbot.llava_flag}
        elif request.feature == "speech":
            chatbot.speech(not chatbot.leap_flag, not chatbot.listen_flag)
            return {"speech_status": not chatbot.leap_flag}
        elif request.feature == "latex":
            chatbot.latex(not chatbot.latex_flag)
            return {"latex_status": chatbot.latex_flag}
        else:
            raise HTTPException(status_code=400, detail="Invalid feature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available_models")
async def get_available_models():
    try:
        models = chatbot.ollama_command_instance.ollama_list()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_command")
async def execute_command(request: CommandRequest):
    try:
        result = chatbot.command_select(request.command)
        response = f"Command executed: {request.command}, Result: {result}"
        await manager.broadcast(response)
        return {"response": response}
    except Exception as e:
        error_message = f"Error executing command: {str(e)}"
        await manager.broadcast(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/available_voices")
async def get_available_voices():
    try:
        fine_tuned_voices, reference_voices = chatbot.get_available_voices()
        return {"fine_tuned_voices": fine_tuned_voices, "reference_voices": reference_voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/select_voice")
async def select_voice(request: dict):
    try:
        chatbot.voice_name = request["voice_name"]
        chatbot.voice_type = request["voice_type"]
        return {"message": f"Voice changed to {chatbot.voice_name} (Type: {chatbot.voice_type})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2020)