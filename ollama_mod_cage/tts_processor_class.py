""" tts_processor.py
     copy paste model names:
        borch/llama3_speed_chat_2
        borch_llama3_speed_chat
        borch_llama3_speed_chat
        c3po
        Jesus 
        borch_llama3po 
        dolphin-llama3 
        dolphin-mistral 
        gemma 
        llama3 
        mistral 
        tic_tac
        
    A class for processing the response sentences and audio generation for the ollama_chat_bot_class
"""

import sounddevice as sd
import soundfile as sf
import threading
import os
import torch
import re
import queue
from TTS.api import TTS
import speech_recognition as sr
from directory_manager_class import directory_manager_class
import numpy as np
import scipy.io.wavfile as wav

class tts_processor_class:
    def __init__(self):
        """a method for initializing the class
        """
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_voice_ref_wav_pack_path = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\active_group\\Public_Voice_Reference_Pack")
        self.conversation_library = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\conversation_library")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.audio_queue = queue.Queue()

    def get_audio(self):
        """ a method for collecting the audio from the microphone
            args: none
            returns: audio
        """
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
        return audio
    
    def recognize_speech(self, audio):
        """ a method for calling the speech recognizer
        """
        speech_str = sr.Recognizer().recognize_google(audio)
        print(f">>{speech_str}<<")
        return speech_str
    
    def process_tts_responses(self, response, voice_name):
        """a method for managing the response preprocessing methods
            args: response
            returns: none
        """
        # Call Sentence Splitter
        tts_response_sentences = self.split_into_sentences(response)
        self.generate_play_audio_loop(tts_response_sentences, voice_name)
        return

    def play_audio_thread(self):
        """A separate thread for audio playback."""
        while True:
            audio_data, sample_rate = self.audio_queue.get()
            sd.play(audio_data, sample_rate)
            sd.wait()

    def generate_audio(self, sentence, voice_name_path, ticker):
        """ a method to generate the audio for the chatbot
            args: sentence, voice_name_path, ticker
            returns: none
        """
        # Generate TTS audio (replace with your actual TTS logic)
        print("starting speech generation:")
        tts_audio = self.tts.tts(text=sentence, speaker_wav=voice_name_path, language="en", speed=3)

        # Convert to NumPy array (adjust dtype as needed)
        tts_audio = np.array(tts_audio, dtype=np.float32)

        # Add the audio data to the queue
        self.audio_queue.put((tts_audio, 22050))

    def generate_play_audio_loop(self, tts_response_sentences, voice_name):
        """ a method to generate and play the audio for the chatbot
            args: tts_sentences
            returns: none
        """
        ticker = 0  # Initialize ticker
        voice_name_path = os.path.join(self.tts_voice_ref_wav_pack_path, f"{voice_name}\\clone_speech.wav")

        # Start the audio playback thread
        audio_thread = threading.Thread(target=self.play_audio_thread)
        audio_thread.start()

        for sentence in tts_response_sentences:
            ticker += 1

            # Generate the audio in a separate thread
            audio_thread = threading.Thread(target=self.generate_audio, args=(sentence, voice_name_path, ticker))
            audio_thread.start()

            # Wait for the audio generation to finish before moving on to the next sentence
            audio_thread.join()

    def split_into_sentences(self, text: str) -> list[str]:
        """A method for splitting the LLAMA response into sentences.
        Args:
            text (str): The input text.
        Returns:
            list[str]: List of sentences.
        """
        # Add spaces around punctuation marks for consistent splitting
        text = " " + text + " "
        text = text.replace("\n", " ")

        # Handle common abbreviations and special cases
        text = re.sub(r"(Mr|Mrs|Ms|Dr|i\.e)\.", r"\1<prd>", text)
        text = re.sub(r"\.\.\.", r"<prd><prd><prd>", text)

        # Split on period, question mark, exclamation mark, or colon followed by optional spaces
        sentences = re.split(r"(?<=\d\.)\s+|(?<=[.!?:])\s+", text)

        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # Combine the number with its corresponding sentence
        combined_sentences = []
        i = 0
        while i < len(sentences):
            if re.match(r"^\d+\.", sentences[i]):
                combined_sentences.append(f"{sentences[i]} {sentences[i + 1]}")
                i += 2
            else:
                combined_sentences.append(sentences[i])
                i += 1

        return combined_sentences
    
    def file_name_voice_filter(self, input):
        """ a method for preprocessing the voice recognition with a filter before forwarding the agent file names.
            args: user_input_agent_name
            returns: user_input_agent_name
        """
        # Use regex to replace all spaces with underscores
        output = re.sub(' ', '_', input).lower()
        return output
    
    def file_name_conversation_history_filter(self, input):
        """ a method for preprocessing the voice recognition with a filter before forwarding the agent file names.
            args: user_input_agent_name
            returns: user_input_agent_name
        """
        # Use regex to replace all spaces with underscores and convert to lowercase
        output = re.sub(' ', '_', input).lower()
        return output