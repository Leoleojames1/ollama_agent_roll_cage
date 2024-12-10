from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio
import ollama
import numpy as np
import base64
import sounddevice as sd
import speech_recognition as sr
from typing import Dict, Any
import os
import json

from ollama_chatbot_wizard import ollama_chatbot_base
from wizard_spell_book.Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_buffer = np.array([])
        
    async def process_audio_data(self, raw_data: bytes) -> np.ndarray:
        """Convert raw audio bytes to numpy array"""
        return np.frombuffer(raw_data, dtype=np.float32)

    def create_audio_data(self, audio_buffer: np.ndarray) -> sr.AudioData:
        """Create AudioData object from numpy array"""
        return sr.AudioData(audio_buffer.tobytes(), self.sample_rate, 2)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.chatbot_states: Dict[str, Dict[str, Any]] = {}
        self.audio_processor = AudioProcessor()
        self.logger = logging.getLogger(__name__)  # Add this line

    async def process_message(self, websocket: WebSocket, agent_id: str, data: Dict):
        """Process incoming WebSocket messages with complete chatbot functionality"""
        if agent_id not in self.chatbot_states:
            raise HTTPException(status_code=404, detail="Agent not found")
            
        state = self.chatbot_states[agent_id]
        chatbot = state["chatbot"]
        
        try:
            if data["type"] == "chat":
                content = data["content"]
                
                # First send back the user's message to update chat history
                await websocket.send_json({
                    "type": "chat_message",
                    "content": content,
                    "role": "user"
                })
                
                # Process message with ollama
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        chatbot.send_prompt,
                        content
                    )
                    
                    if response:
                        # Ensure response is a string
                        response_str = str(response) if response is not None else ""
                        logger.info(f"Sending response: {response_str}")
                        
                        # Send the response
                        await websocket.send_json({
                            "type": "chat_response",
                            "content": response_str,
                            "role": "assistant",
                            "is_stream": False
                        })
                        
                        # Save chat history
                        if hasattr(chatbot, 'save_to_json'):
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                chatbot.save_to_json,
                                chatbot.save_name,
                                chatbot.user_input_model_select
                            )
                    
                except Exception as e:
                    logger.error(f"Error in message processing: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing message: {str(e)}",
                        "role": "system"
                    })
                
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            await websocket.send_json({
                "type": "error",
                "content": str(e),
                "role": "system"
            })
            
    async def process_text_message(self, chatbot, content: str):
        """Process text messages through the chatbot"""
        try:
            logger.info(f"Sending prompt to model: {content}")
            response = await asyncio.get_event_loop().run_in_executor(
                None, chatbot.send_prompt, content
            )
            return response
        except Exception as e:
            logger.error(f"Error in process_text_message: {str(e)}")
            raise

    async def connect(self, websocket: WebSocket, agent_id: str):
        """Connect a new client and initialize their chatbot"""
        try:
            await websocket.accept()
            
            # Guard against duplicate connections
            if agent_id in self.active_connections:
                await self.disconnect(agent_id)
                
            self.active_connections[agent_id] = websocket
            
            # Create new chatbot instance with error handling
            try:
                chatbot = ollama_chatbot_base()
                chatbot_instance = await self.initialize_chatbot(chatbot)
                
                # Store chatbot state - This was missing before
                self.chatbot_states[agent_id] = {
                    "chatbot": chatbot_instance,
                    "audio_buffer": np.array([])
                }
                
                # Send initial connection success message
                await websocket.send_json({
                    "type": "connection_status",
                    "content": "Connected successfully",
                    "agent_id": agent_id
                })
                
            except Exception as e:
                logger.error(f"Failed to initialize chatbot: {e}")
                await websocket.close()
                if agent_id in self.active_connections:
                    del self.active_connections[agent_id]
                raise
                
        except Exception as e:
            logger.error(f"Error in connect: {e}")
            if agent_id in self.active_connections:
                del self.active_connections[agent_id]
            raise

    async def disconnect(self, agent_id: str):
        """Safely disconnect a client and clean up its state"""
        try:
            if agent_id in self.active_connections:
                websocket = self.active_connections[agent_id]
                try:
                    await websocket.close()
                except Exception as e:
                    self.logger.error(f"Error closing websocket: {e}")
                del self.active_connections[agent_id]
                    
            if agent_id in self.chatbot_states:
                chatbot = self.chatbot_states[agent_id]["chatbot"]
                if hasattr(chatbot, 'cleanup'):
                    await asyncio.get_event_loop().run_in_executor(None, chatbot.cleanup)
                del self.chatbot_states[agent_id]
                    
            self.logger.info(f"Client disconnected: {agent_id}")
                
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    async def initialize_chatbot(self, chatbot):
        """Initialize chatbot synchronously"""
        try:
            # Basic initialization
            chatbot.initializeBasePaths()
            chatbot.initializeAgent()
            chatbot.initializeChat()
            chatbot.initializeCommandLibrary()
            
            # Set default flags and model
            chatbot.AGENT_FLAG = False
            chatbot.MEMORY_CLEAR_FLAG = False
            chatbot.user_input_model_select = None
            chatbot.TTS_FLAG = False
            
            # Initialize TTS processor if needed
            if not hasattr(chatbot, 'tts_processor_instance'):
                chatbot.tts_processor_instance = None
            
            # Initialize components
            chatbot.initializeSpells()
            chatbot.createAgentDict()
            
            logger.info("Chatbot initialized successfully")
            logger.info(f"Using model: {chatbot.user_input_model_select}")
            
            return chatbot
                
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise

    async def stream_response(self, websocket: WebSocket, response_generator):
        try:
            async for chunk in response_generator:
                # Ensure content is always a string before sending
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    # Convert content to string if it's a dict
                    if isinstance(content, dict):
                        content = json.dumps(content)
                    else:
                        content = str(content)
                        
                    await websocket.send_json({
                        "type": "chat_response",
                        "content": content,
                        "role": "assistant",
                        "is_stream": True
                    })
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            await websocket.send_json({
                "type": "error",
                "content": "Error streaming response",
                "role": "system"
            })
          
    async def cleanup_resources(self, agent_id: str):
        """Cleanup all resources for an agent"""
        try:
            if agent_id in self.chatbot_states:
                state = self.chatbot_states[agent_id]
                
                # Cleanup audio processing
                if state.get("audio_buffer") is not None:
                    state["audio_buffer"] = None
                    
                # Cleanup chatbot resources
                if state.get("chatbot"):
                    chatbot = state["chatbot"]
                    if hasattr(chatbot, 'cleanup'):
                        await asyncio.get_event_loop().run_in_executor(
                            None, chatbot.cleanup
                        )
                        
                del self.chatbot_states[agent_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up resources for {agent_id}: {e}")
          
    async def handle_error(self, websocket: WebSocket, error: Exception, context: str = ""):
        """Centralized error handling"""
        error_message = f"Error in {context}: {str(error)}"
        logger.error(error_message)
        
        try:
            await websocket.send_json({
                "type": "error",
                "content": error_message,
                "role": "system"
            })
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
        
    async def monitor_connection(self, agent_id: str):
        """Monitor connection health and cleanup if needed"""
        while True:
            try:
                if agent_id not in self.active_connections:
                    break
                    
                websocket = self.active_connections[agent_id]
                try:
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    await self.disconnect(agent_id)
                    break
                    
            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                await self.disconnect(agent_id)
                break
        
    async def sync_command_library(self):
        """Sync command library across all instances"""
        try:
            base_chatbot = ollama_chatbot_base()
            commands = list(base_chatbot.command_library.keys())
            
            for state in self.chatbot_states.values():
                chatbot = state["chatbot"]
                chatbot.command_library = base_chatbot.command_library.copy()
                
            return commands
        except Exception as e:
            logger.error(f"Error syncing command library: {e}")
            raise
        
    async def execute_command(self, websocket: WebSocket, agent_id: str, command: str):
        """Execute a command through the chatbot instance"""
        try:
            if agent_id not in self.chatbot_states:
                raise HTTPException(status_code=404, detail="Agent not found")
                
            chatbot = self.chatbot_states[agent_id]["chatbot"]
            result = await asyncio.get_event_loop().run_in_executor(
                None, chatbot.command_select, command
            )
            
            # Format the command result consistently
            formatted_result = {
                "success": isinstance(result, bool) and result,  # Handle boolean results
                "command": command,
                "result": str(result) if result is not None else "Command executed"
            }

            # Convert to string if it's a dict
            result_content = json.dumps(formatted_result) if isinstance(formatted_result, dict) else str(formatted_result)
            
            await websocket.send_json({
                "type": "command_result",
                "content": result_content,
                "role": "system"
            })
            
            return formatted_result
                
        except Exception as e:
            error_msg = f"Error in execute_command: {str(e)}"
            logger.error(error_msg)
            await websocket.send_json({
                "type": "error",
                "content": error_msg,
                "role": "system"
            })
            return {
                "success": False,
                "command": command,
                "error": error_msg
            }
            
    async def get_voice_state(self, agent_id: str):
        """Get current voice state for an agent"""
        if agent_id in self.chatbot_states:
            chatbot = self.chatbot_states[agent_id]["chatbot"]
            return {
                "tts_enabled": chatbot.TTS_FLAG,
                "current_voice": {
                    "type": chatbot.voice_type,
                    "name": chatbot.voice_name
                }
            }
        return None
    
    async def process_audio(self, websocket: WebSocket, agent_id: str, audio_data: np.ndarray):
        """Process audio data and handle transcription/response"""
        state = self.chatbot_states[agent_id]
        chatbot = state["chatbot"]
        
        state["audio_buffer"] = np.concatenate([state["audio_buffer"], audio_data])
        
        if len(state["audio_buffer"]) >= self.audio_processor.sample_rate:
            audio = self.audio_processor.create_audio_data(state["audio_buffer"])
            text = await self.process_audio_message(chatbot, audio)
            
            if text:
                await websocket.send_json({
                    "type": "transcription",
                    "content": text
                })
                
                if not text.startswith('/'):
                    response = await self.process_text_message(chatbot, text)
                    await websocket.send_json({
                        "type": "response",
                        "content": response
                    })
                    
                    if chatbot.TTS_FLAG:
                        audio = await self.process_tts(chatbot, response)
                        if audio is not None:
                            await self.broadcast_audio(agent_id, audio)
                else:
                    result = await self.process_command(chatbot, text)
                    await websocket.send_json({
                        "type": "command_result",
                        "content": result
                    })
                    
            state["audio_buffer"] = np.array([])

    async def process_audio_message(self, chatbot, audio: sr.AudioData):
        """Process audio data through speech recognition"""
        return await asyncio.get_event_loop().run_in_executor(
            None, chatbot.speech_recognizer_instance.recognize_speech, audio
        )

    async def process_command(self, chatbot, command: str):
        """Process chatbot commands"""
        return await asyncio.get_event_loop().run_in_executor(
            None, chatbot.command_select, command
        )

    async def process_tts(self, chatbot, text: str):
        """Process text-to-speech"""
        return await asyncio.get_event_loop().run_in_executor(
            None, chatbot.tts_processor_instance.process_tts_responses,
            text, chatbot.voice_name
        )

    async def broadcast_audio(self, agent_id: str, audio_data: np.ndarray):
        """Broadcast audio data to connected clients"""
        for id_, connection in self.active_connections.items():
            if id_ != agent_id:
                await connection.send_bytes(audio_data.tobytes())

    async def set_model(self, agent_id: str, model_name: str):
        """Set model for a specific agent"""
        if agent_id in self.chatbot_states:
            try:
                state = self.chatbot_states[agent_id]
                chatbot = state["chatbot"]
                chatbot.set_model(model_name)
                return {"status": "success", "model": model_name}
            except Exception as e:
                logger.error(f"Error setting model for agent {agent_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    async def toggle_whisper(self, agent_id: str, enable: bool, model_size: str = "base"):
        """Toggle Whisper speech recognition"""
        if agent_id in self.chatbot_states:
            chatbot = self.chatbot_states[agent_id]["chatbot"]
            if enable:
                success = chatbot.speech_recognizer_instance.enable_whisper(model_size)
                return {"status": "enabled" if success else "failed"}
            else:
                chatbot.speech_recognizer_instance.disable_whisper()
                return {"status": "disabled"}
        return {"status": "error", "message": "Agent not found"}

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = ConnectionManager()
    yield
    for agent_id in list(manager.active_connections.keys()):
        await manager.disconnect(agent_id)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """Main websocket endpoint for text communication"""
    try:
        await manager.connect(websocket, agent_id)
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Received message from {agent_id}: {data}")
                
                # Validate message format
                if "type" not in data or "content" not in data:
                    raise ValueError("Invalid message format - missing type or content")

                elif data["type"] == "command":
                    # Use existing chatbot instance to process command
                    if agent_id in manager.chatbot_states:
                        chatbot = manager.chatbot_states[agent_id]["chatbot"]
                        command = data["content"]
                        success = await asyncio.get_event_loop().run_in_executor(
                            None, chatbot.command_select, command
                        )
                        await websocket.send_json({
                            "type": "command_result",
                            "content": {
                                "success": success,
                                "command": command
                            }
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "content": "No active chatbot instance found"
                        })
                elif data["type"] == "get_voices":
                    # Get available voices from existing chatbot instance
                    if agent_id in manager.chatbot_states:
                        chatbot = manager.chatbot_states[agent_id]["chatbot"]
                        fine_tuned_voices, reference_voices = await asyncio.get_event_loop().run_in_executor(
                            None, chatbot.get_available_voices
                        )
                        await websocket.send_json({
                            "type": "voices",
                            "content": {
                                "fine_tuned": fine_tuned_voices,
                                "reference": reference_voices
                            }
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "content": "No active chatbot instance found"
                        })
                elif data["type"] == "set_voice":
                    # Set voice for existing chatbot instance
                    if agent_id in manager.chatbot_states:
                        chatbot = manager.chatbot_states[agent_id]["chatbot"]
                        voice_type = data["content"].get("type")
                        voice_name = data["content"].get("name")
                        
                        if voice_type and voice_name:
                            await asyncio.get_event_loop().run_in_executor(
                                None, 
                                chatbot.set_voice,
                                voice_type,
                                voice_name
                            )
                            await websocket.send_json({
                                "type": "voice_set",
                                "content": {
                                    "type": voice_type,
                                    "name": voice_name
                                }
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "content": "Voice type and name required"
                            })
                else:
                    # Process other message types (chat, etc.)
                    await manager.process_message(websocket, agent_id, data)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for agent {agent_id}")
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send_json({
                    "type": "error", 
                    "content": "Invalid message format"
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": str(e)
                })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(agent_id)


@app.websocket("/audio-stream/{agent_id}")
async def audio_websocket_endpoint(websocket: WebSocket, agent_id: str):
    """Audio streaming websocket endpoint"""
    await manager.connect(websocket, agent_id)
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_data = await manager.audio_processor.process_audio_data(data)
            
            state = manager.chatbot_states[agent_id]
            recognizer = state["chatbot"].speech_recognizer_instance
            
            if not recognizer.is_listening:
                if await recognizer.wait_for_wake_word():
                    await websocket.send_json({
                        "type": "status",
                        "content": "Wake word detected"
                    })
                continue
            
            text = await manager.process_audio_message(recognizer, audio_data)
            if text:
                await websocket.send_json({
                    "type": "transcription",
                    "content": text
                })
    except WebSocketDisconnect:
        await manager.disconnect(agent_id)

@app.post("/set_model")
async def set_model(model_data: dict):
    """Set active model endpoint"""
    try:
        model_name = model_data.get('model')
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        for state in manager.chatbot_states.values():
            chatbot = state["chatbot"]
            chatbot.set_model(model_name)
        
        logger.info(f"Model set to: {model_name}")
        return {"status": "success", "model": model_name}
    except Exception as e:
        logger.error(f"Error setting model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/command_library")
async def get_command_library():
    """Get available commands endpoint"""
    try:
        # Create temporary chatbot instance to get commands
        chatbot = ollama_chatbot_base()
        commands = list(chatbot.command_library.keys())
        return {"commands": commands}
    except Exception as e:
        logger.error(f"Error getting command library: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Update the get_models endpoint
@app.get("/available_models")
async def get_models():
    """Get available models endpoint"""
    try:
        # Create command instance with paths
        cmd = ollama_commands(None, {
            'current_dir': os.getcwd(),
            'parent_dir': os.path.dirname(os.getcwd())
        })
        models = await cmd.ollama_list()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/toggle_whisper/{agent_id}")
async def toggle_whisper(agent_id: str, enable: bool, model_size: str = "base"):
    """Toggle Whisper endpoint"""
    return await manager.toggle_whisper(agent_id, enable, model_size)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2020)