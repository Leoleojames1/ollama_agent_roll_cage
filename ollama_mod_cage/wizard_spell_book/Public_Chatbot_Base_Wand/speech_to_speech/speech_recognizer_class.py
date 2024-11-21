# In speech_recognizer_class.py
import pyaudio
import speech_recognition as sr
import numpy as np
import whisper
import json
import threading
import queue
import audioop
import wave
import tempfile
import os

class speech_recognizer_class:
    def __init__(self, colors, chunk_flag, listen_flag, auto_speech_flag):
        """Initialize speech recognizer with queues and improved wake word detection"""
        # Basic flags and settings
        self.colors = colors
        self.auto_speech_flag = False
        self.chunk_flag = False
        self.listen_flag = False
        self.is_listening = True
        self.is_active = True
        
        # Voice settings
        self.wake_word = "Yo Jamie"  # Configurable wake word
        self.recognizer = sr.Recognizer()
        self.use_wake_commands = False
        
        # Default to Google Speech Recognition
        self.use_whisper = False
        self.whisper_model = None
        
        # Queue system for audio processing
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Audio settings
        self.FORMAT = pyaudio.paInt32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 2
        self.SILENCE_THRESHOLD = 605
        self.SILENCE_DURATION = 0.25
        
        # Audio buffers
        self.audio_data = np.array([])
        self.audio_buffer = []
        self.frames = []
        self.is_recording = False

    def enable_whisper(self, model_size="tiny"):
        """Optional method to enable and load Whisper with minimal model"""
        try:
            import whisper
            self.use_whisper = True
            self.whisper_model = whisper.load_model(model_size)
            print(f"Whisper {model_size} model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.use_whisper = False
            self.whisper_model = None
            return False

    def listen(self, threshold=605, silence_duration=0.25):
        """Enhanced listen method with silence detection"""
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except IOError:
            print(self.colors["RED"] + "Error: Could not access the microphone." + self.colors["END"])
            audio.terminate()
            return None

        frames = []
        silent_frames = 0
        sound_detected = False

        while True:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                rms = audioop.rms(data, 2)

                if rms > threshold:
                    silent_frames = 0
                    sound_detected = True
                else:
                    silent_frames += 1

                if sound_detected and (silent_frames * (self.CHUNK / self.RATE) > silence_duration):
                    break

            except Exception as e:
                print(f"Error during recording: {e}")
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if sound_detected:
            # Save to temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            return temp_file.name
        return None

    def recognize_speech(self, audio_input):
        """Enhanced speech recognition with multiple backends"""
        if isinstance(audio_input, str):
            # Input is a file path
            if self.use_whisper and self.whisper_model is not None:
                try:
                    transcript = self.whisper_model.transcribe(audio_input)
                    speech_str = transcript["text"]
                except Exception as e:
                    print(f"Whisper error: {e}")
                    speech_str = self.recognize_with_google(audio_input)
            else:
                speech_str = self.recognize_with_google(audio_input)
        else:
            # Input is audio data
            speech_str = self.recognize_with_google(audio_input)
            
        print(self.colors["GREEN"] + f"<<<👂 SPEECH RECOGNIZED 👂 >>> " + self.colors["OKBLUE"] + f"{speech_str}")
        return speech_str

    def recognize_with_google(self, audio_input):
        """Use Google Speech Recognition"""
        try:
            if isinstance(audio_input, str):
                # Convert file to AudioData
                with sr.AudioFile(audio_input) as source:
                    audio_data = self.recognizer.record(source)
            else:
                audio_data = audio_input
                
            return self.recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition; {e}"

    def wait_for_wake_word(self):
        """Wait for wake word activation"""
        while True:
            temp_file = self.listen()
            if temp_file:
                try:
                    speech_text = self.recognize_speech(temp_file).lower()
                    
                    if self.wake_word in speech_text:
                        print(self.colors["OKBLUE"] + "Wake word detected! Starting to listen..." + self.colors["END"])
                        self.is_listening = True
                        return True
                        
                finally:
                    # Cleanup temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        return False

    def start_continuous_listening(self):
        """Start continuous listening process"""
        self.is_listening = True
        while self.is_listening:
            temp_file = self.listen()
            if temp_file:
                try:
                    speech_text = self.recognize_speech(temp_file)
                    self.text_queue.put(speech_text)
                    
                    # Check for sleep commands
                    if any(phrase in speech_text.lower() for phrase in [
                        "thanks alexa", "thank you alexa", "okay, thanks alexa"
                    ]):
                        print(self.colors["OKBLUE"] + "Sleep word detected! Waiting for wake word..." + self.colors["END"])
                        self.is_listening = False
                        break
                        
                finally:
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def get_next_text(self):
        """Get next text from queue"""
        try:
            return self.text_queue.get_nowait()
        except queue.Empty:
            return None

    def cleanup(self):
        """Clean up resources"""
        self.is_listening = False
        self.is_active = False
        if self.whisper_model is not None:
            self.whisper_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def set_wake_word(self, wake_word: str):
        """Set new wake word"""
        self.wake_word = wake_word.lower()