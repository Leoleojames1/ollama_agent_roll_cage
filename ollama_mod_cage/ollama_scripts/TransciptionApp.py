import threading
import whisper
import pyaudio
import wave
import tkinter as tk
from tkinter import scrolledtext
import tempfile
import audioop
import os
from tkinter import ttk
import queue
import ollama
from TTS.api import TTS
import pygame
import io

os.environ["PATH"] += os.pathsep + r"D:\CodingGit_StorageHDD\ffmpeg_LIBRARY\ffmpeg-master-latest-win64-gpl-shared\bin"

class TranscriptionApp(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.root.title("bot1")
        
        self.text_queue = queue.Queue()
        self.speech_queue = queue.Queue()
        self.is_listening = True
        self.is_active = True
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=24000)  # Higher frequency for faster playback
        
        # Initialize TTS model
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            print("CUDA-compatible GPU is not available. Using CPU instead.")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        # Start speech processing thread
        threading.Thread(target=self.process_speech_queue, daemon=True).start()
        
        # Add search frame
        self.search_frame = tk.Frame(root, bg='black')
        self.search_frame.pack(fill=tk.X)
        
        self.search_entry = tk.Entry(
            self.search_frame,
            bg='black',
            fg='white',
            font=('Arial', 12),
            insertbackground='white'
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15, pady=15)
        self.search_entry.bind('<Return>', lambda event: self.search_text())
        
        self.search_button = tk.Button(
            self.search_frame,
            text="Search",
            command=self.search_text,
            bg='black',
            fg='white',
            font=('Arial', 12)
        )
        self.search_button.pack(side=tk.RIGHT, padx=15, pady=15)
        
        self.text_box = scrolledtext.ScrolledText(
            root, 
            width=50, 
            height=50,
            bg='black',
            fg='white',
            font=('Arial', 17),
            insertbackground='black',
            takefocus=True,
        )
        self.text_box.pack(fill=tk.BOTH, expand=True)

        self.model = whisper.load_model("tiny")
        
        # Start the recording and transcribing thread
        threading.Thread(target=self.wait_for_wake_word, daemon=True).start()
        # Start the UI update thread
        self.process_queue()

    def speak_text(self, text, lang='en'):
        if text.strip():
            self.speech_queue.put((text, lang))

    def process_speech_queue(self):
        while True:
            try:
                text, lang = self.speech_queue.get()
                try:
                    # Use Coqui TTS to generate audio
                    audio = self.tts.tts(text, language="en")
                    
                    # Load and play the audio
                    pygame.mixer.music.unload()  # Clear any previous audio
                    pygame.mixer.Sound(audio).play()
                    
                    # Wait for the audio to finish playing
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(-1)
                        
                except Exception as e:
                    print(f"Speech error: {str(e)}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Queue processing error: {str(e)}")

    def get_llama_response(self, text):
        if "thanks alexa" in text.lower() or "thank you alexa" in text.lower() or "okay, thanks alexa" in text.lower():
            self.is_listening = False
            response_text = "It was my pleasure to assist you. If you need anything else, just say 'Alexa' to wake me up. Have a great day!"
            self.text_queue.put(response_text)
            self.speak_text(response_text)
        else:
            try:
                stream = ollama.chat(
                    model='llama3.2:1b',
                    messages=[{'role': 'user', 'content': text}],
                    stream=True,
                )

                sentence_buffer = ""
                for chunk in stream:
                    if 'message' in chunk and 'message' in chunk:
                        content = chunk['message']['content']
                        self.text_queue.put(content)
                        
                        sentence_buffer += content
                        
                        # Check for sentence endings
                        while any(end in sentence_buffer for end in '.!?'):
                            # Find the first sentence ending
                            end_indices = [sentence_buffer.find(end) for end in '.!?' if end in sentence_buffer]
                            first_end = min(i for i in end_indices if i != -1)
                            
                            # Extract the complete sentence
                            complete_sentence = sentence_buffer[:first_end + 1].strip()
                            if complete_sentence:
                                self.speak_text(complete_sentence)
                            
                            # Keep the remainder in the buffer
                            sentence_buffer = sentence_buffer[first_end + 1:].strip()
                
                # Speak any remaining text
                if sentence_buffer.strip():
                    self.speak_text(sentence_buffer)
                    
            except Exception as e:
                error_message = f"Error getting LLM response: {str(e)}\n"
                self.text_queue.put(error_message)
                return error_message

    def wait_for_wake_word(self):
        while True:
            temp_file = self.listen()
            if temp_file:
                transcript = self.transcribe(temp_file)
                text = transcript['text'].lower()
                
                if 'alexa' in text:
                    self.text_queue.put("Wake word detected! Starting to listen...\n")
                    self.speak_text("Hello, how can I help you?")
                    self.is_listening = True
                    self.record_and_transcribe()
                
                os.remove(temp_file)

    def search_text(self):
        search_term = self.search_entry.get().lower()
        if not search_term:
            return
            
        # Remove previous highlights
        self.text_box.tag_remove('search', '1.0', tk.END)
        
        # Search and highlight
        start_pos = '1.0'
        while True:
            start_pos = self.text_box.search(search_term, start_pos, tk.END, nocase=True)
            if not start_pos:
                break
            end_pos = f"{start_pos}+{len(search_term)}c"
            self.text_box.tag_add('search', start_pos, end_pos)
            start_pos = end_pos
        
        # Configure highlight color
        self.text_box.tag_config('search', background='yellow', foreground='black')
        
    def listen(self, threshold=605, silence_duration=0.25):
        FORMAT = pyaudio.paInt32      
        CHANNELS = 1
        RATE = 44100
        CHUNK = 2

        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except IOError:
            print("Error: Could not access the microphone.")
            audio.terminate()
            return None

        frames = []
        silent_frames = 0
        sound_detected = False

        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            rms = audioop.rms(data, 2)

            if rms > threshold:
                silent_frames = 0
                sound_detected = True
            else:
                silent_frames += 1

            if sound_detected and (silent_frames * (CHUNK / RATE) > silence_duration):
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if sound_detected:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            return temp_file.name
        return None

    def transcribe(self, temp_file):
        if temp_file:
            transcript = self.model.transcribe(temp_file)
            try:
                return transcript
            except:
                pass
        return {"text": "", "language": "unknown"}

    def detectlang(self, transcript):
        detected_language = transcript.get('language', 'unknown')
        return detected_language

    def update_text_box(self, text_data):
        self.text_box.insert(tk.END, text_data)
        self.text_box.see(tk.END)

    def process_queue(self):
        try:
            while True:
                text_data = self.text_queue.get_nowait()
                self.update_text_box(text_data)
        except queue.Empty:
            pass
        finally:
            self.root.after(10, self.process_queue)

    def record_and_transcribe(self):
        while self.is_listening:
            temp_file = self.listen()
            if temp_file:
                transcript = self.transcribe(temp_file)
                lang = self.detectlang(transcript)

                output_text = f"{lang}\n{transcript['text']}\n-----------------------\n\n"
                self.text_queue.put(output_text)
                
                if "thanks alexa" in transcript['text'].lower() or "thank you alexa" in transcript['text'].lower() or "okay, thanks alexa" in transcript['text'].lower():
                    self.is_listening = False
                    self.text_queue.put("Sleep word detected! Waiting for wake word...\n")
                    break
                
                # Process LLM response in a separate thread
                def process_llm_response():
                    self.text_queue.put("LLM Response:\n")
                    self.get_llama_response(transcript['text'])
                    self.text_queue.put("\n" + "-"*50 + "\n")

                threading.Thread(target=process_llm_response, daemon=True).start()
                os.remove(temp_file)

def main():
    root = tk.Tk()
    root.configure(bg='black')
    root.attributes('-alpha', 0.7)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.minsize(100, 100)
    
    style = ttk.Style()
    style.configure("Custom.TLabel", foreground="black", background="black", font=('Arial', 19))
    
    app = TranscriptionApp(root)
    app.pack(fill=tk.BOTH, expand=True)
    
    def start_move(event):
        root.x = event.x
        root.y = event.y

    def do_move(event):
        deltax = event.x - root.x
        deltay = event.y - root.y
        x = root.winfo_x() + deltax
        y = root.winfo_y() + deltay
        root.geometry(f"+{x}+{y}")

    root.bind("<Button-1>", start_move)
    root.bind("<B1-Motion>", do_move)
    
    root.mainloop()

if __name__ == "__main__":
    main()