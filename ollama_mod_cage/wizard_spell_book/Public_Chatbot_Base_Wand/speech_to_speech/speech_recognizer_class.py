import pyaudio
import speech_recognition as sr
import numpy as np
import whisper
import json
import threading

# -------------------------------------------------------------------------------------------------
class speech_recognizer_class:
    def __init__(self, colors):
        self.colors = colors
        self.auto_speech_flag = False
        self.chunk_flag = False
        self.listen_flag = False
        self.wake_word = "Yo Jaime"
        self.recognizer = sr.Recognizer()
        self.use_wake_commands = False
        self.use_whisper = True
        self.whisper_model = whisper.load_model("turbo") if self.use_whisper else None
        self.audio_data = np.array([])
        self.audio_buffer = []  # Buffer to store incoming audio chunks
        self.is_recording = False
    
    # -------------------------------------------------------------------------------------------------
    async def process_audio_stream(self, websocket):
        """Process incoming audio stream from WebSocket"""
        self.is_recording = True
        try:
            while self.is_recording:
                try:
                    data = await websocket.receive_bytes()
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    self.audio_data = np.concatenate([self.audio_data, audio_chunk])
                    self.audio_buffer.append(data)
                    
                    # Check if we have enough audio data to process
                    if len(self.audio_buffer) >= 32:  # About 1 second of audio at 32 chunks
                        audio_data = b''.join(self.audio_buffer)
                        self.audio_buffer = []  # Clear buffer
                        
                        # Convert to AudioData object for recognition
                        audio = sr.AudioData(audio_data, 16000, 2)
                        
                        # Process with selected recognition method
                        if self.use_whisper:
                            audio_np = np.frombuffer(audio.frame_data, dtype=np.int16)
                            audio_np = audio_np.astype(np.float32) / 32768.0
                            result = self.whisper_model.transcribe(audio_np)
                            text = result["text"]
                        else:
                            text = self.recognizer.recognize_google(audio)
                            
                        # Send recognition result back to client
                        await websocket.send_json({
                            "type": "recognition_result",
                            "text": text
                        })
                        
                        # Send audio visualization data
                        await websocket.send_json({
                            "type": "audio_data",
                            "data": self.audio_data.tolist()[-1024:]  # Send last 1024 samples for visualization
                        })
                        
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.is_recording = False
            self.audio_data = np.array([])
            self.audio_buffer = []
            
    # -------------------------------------------------------------------------------------------------
    def start_recording(self):
        """Initialize recording state"""
        self.is_recording = True
        self.frames = []
        self.audio_data = np.array([])
    
    # ------------------------------------------------------------------------------------------------- 
    def stop_recording(self):
        """Clean up recording state"""
        self.is_recording = False
        self.frames = []
            
    # -------------------------------------------------------------------------------------------------
    def get_audio(self):
        """ a method for getting the user audio from the microphone
            args: none
        """
        print(self.colors["OKBLUE"] + f">>> AUDIO RECORDING <<<")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []

        while self.auto_speech_flag and not self.chunk_flag:
            data = stream.read(1024)
            frames.append(data)

        print(self.colors["OKBLUE"] + f">>> AUDIO RECEIVED <<<")
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert the audio data to an AudioData object
        audio = sr.AudioData(b''.join(frames), 16000, 2)
        self.chunk_flag = False  # Set chunk_flag to False here to indicate that the audio has been received
        return audio
    
    # -------------------------------------------------------------------------------------------------
    def get_audio_data(self):
        return self.audio_data
    
    # -------------------------------------------------------------------------------------------------
    def recognize_speech(self, audio):
        """ a method for calling the speech recognizer
            args: audio
            returns: speech_str
        """
        if self.use_whisper:
            # Use Whisper for speech recognition
            audio_np = np.frombuffer(audio.frame_data, dtype=np.int16)
            audio_np = audio_np.astype(np.float32) / 32768.0
            speech_str = self.whisper_model.transcribe(audio_np)["text"]
        else:
            # Use Google Speech Recognition as a fallback
            try:
                speech_str = self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                speech_str = "Google Speech Recognition could not understand audio"
            except sr.RequestError as e:
                speech_str = f"Could not request results from Google Speech Recognition; {e}"
        
        print(self.colors["GREEN"] + f"<<<👂 SPEECH RECOGNIZED 👂 >>> " + self.colors["OKBLUE"] + f"{speech_str}")
        return speech_str

    # -------------------------------------------------------------------------------------------------
    def wake_words(self):
        """ A method for recognizing speech with wake word detection. """
        print(self.colors["OKBLUE"] + "Listening for wake word...")
        while True:
            audio = self.get_audio()
            
            # Convert audio data to numpy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            
            # Check if the audio is not silent
            if np.abs(audio_data).mean() > 500:  # Adjust this threshold as needed
                try:
                    speech_str = self.recognize_speech(audio)
                    
                    if speech_str.lower().startswith(self.wake_word.lower()):
                        # Remove the wake word from the speech string
                        actual_command = speech_str[len(self.wake_word):].strip()
                        print(self.colors["GREEN"] + f"Wake word detected! Command: {actual_command}")
                        return actual_command
                    else:
                        print(self.colors["YELLOW"] + "Speech detected, but wake word not found.")
                except sr.UnknownValueError:
                    print(self.colors["YELLOW"] + "Speech was unintelligible.")
                except sr.RequestError as e:
                    print(self.colors["RED"] + f"Could not request results; {e}")
            else:
                print(self.colors["BLUE"] + "Silence detected, continuing to listen...")

    # -------------------------------------------------------------------------------------------------
    def auto_speech_set(self, flag, listen_flag):
        self.auto_speech_flag = flag
        if listen_flag == False:
            self.auto_speech_flag = False
        if not flag:
            print("- speech to text deactivated -")
        print(f"auto_speech_flag FLAG STATE: {self.auto_speech_flag}")

    # -------------------------------------------------------------------------------------------------
    def chunk_speech(self, flag):
        self.chunk_flag = flag
        
    # -------------------------------------------------------------------------------------------------
    def interrupt_speech(self):
        self.auto_speech_flag = False
        self.chunk_flag = False
        
    # -------------------------------------------------------------------------------------------------
    def toggle_wake_commands(self):
        """Toggle wake word detection"""
        self.use_wake_commands = not self.use_wake_commands
        print(f"Wake word detection {'enabled' if self.use_wake_commands else 'disabled'}")
        
    # -------------------------------------------------------------------------------------------------
    def set_wake_word(self, wake_word: str):
        """Set a new wake word"""
        self.wake_word = wake_word
        print(f"Wake word set to: {wake_word}")

    # -------------------------------------------------------------------------------------------------
    def toggle_whisper(self, enable):
        """ Method to toggle Whisper on/off
            args: enable: boolean
            returns: none
        """
        self.use_whisper = enable
        if enable:
            self.whisper_model = whisper.load_model("base")
            print(self.colors["GREEN"] + "Whisper enabled.")
        else:
            self.whisper_model = None
            print(self.colors["RED"] + "Whisper disabled. Google Speech Recognition will be used.")

    # -------------------------------------------------------------------------------------------------
    def get_discord_audio(self, transcribe_to_json=False, save_audio=False):
        """ Method to capture Discord audio and transcribe it using Whisper or Google.
            args: transcribe_to_json: boolean, save_audio: boolean
            returns: none
        """
        print(self.colors["OKBLUE"] + ">>> CAPTURING DISCORD AUDIO <<<")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []

        while self.auto_speech_flag and not self.chunk_flag:
            data = stream.read(1024)
            frames.append(data)

        print(self.colors["OKBLUE"] + f">>> AUDIO RECEIVED <<<")
        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_np = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        if save_audio:
            with open("discord_audio.wav", "wb") as f:
                f.write(b''.join(frames))
        
        if transcribe_to_json:
            if self.use_whisper:
                transcription = self.whisper_model.transcribe(audio_np)["text"]
            else:
                # Use Google Speech Recognition for transcription
                audio = sr.AudioData(b''.join(frames), 16000, 2)
                transcription = self.recognize_speech(audio)

            transcription_data = {"transcription": transcription}
            with open("discord_transcription.json", "w") as json_file:
                json.dump(transcription_data, json_file)
            print(self.colors["GREEN"] + ">>> DISCORD TRANSCRIPTION SAVED TO JSON <<<")

    # -------------------------------------------------------------------------------------------------
    def toggle_discord_audio_recognition(self, enable):
        """ Method to toggle Discord audio recognition on/off
            args: enable: boolean
            returns: none
            #TODO ADD COMMAND CENSOR TO DISCORD AUDIO, ONLY THE ADMIN USER HAS COMMAND ACCESS
        """
        self.auto_speech_flag = enable
        if enable:
            print(self.colors["GREEN"] + "Discord audio recognition enabled.")
        else:
            print(self.colors["RED"] + "Discord audio recognition disabled.")
    
    # -------------------------------------------------------------------------------------------------
    def start_recognition_threads(self, transcribe_to_json=False, save_audio=False):
        """ Method to start recognition of both user and Discord audio simultaneously.
            args: transcribe_to_json: boolean, save_audio: boolean
            returns: none
        """
        print(self.colors["OKBLUE"] + ">>> STARTING AUDIO RECOGNITION THREADS <<<")
        
        user_thread = threading.Thread(target=self.wake_words)
        discord_thread = threading.Thread(target=self.get_discord_audio, args=(transcribe_to_json, save_audio))

        user_thread.start()
        discord_thread.start()

        user_thread.join()
        discord_thread.join()

        print(self.colors["OKBLUE"] + ">>> AUDIO RECOGNITION THREADS COMPLETED <<<")