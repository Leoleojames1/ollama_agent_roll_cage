""" tts_processor.py
     copy paste model names:
        borch/llama3_speed_chat
        c3po
        
    A class for processing the response sentences and audio generation for the ollama_chat_bot_class
"""

import sounddevice as sd
import soundfile as sf
import time
import threading
import os
import torch
import re
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
        self.tts_voice_ref_wav_path = os.path.join(self.parent_dir, "AgentFiles\\Ignored_TTS\\pipeline\\active_group\\clone_speech.wav")
        self.tts_store_wav_locker_path = os.path.join(self.parent_dir, "ollama_mod_cage\\current_speech_wav")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

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
        return sr.Recognizer().recognize_google(audio)
    
    def process_tts_responses(self, response):
        """a method for managing the response preprocessing methods
            args: response
            returns: none
        """
        # Call Sentence Splitter
        tts_response_sentences = self.split_into_sentences(response)
        self.generate_play_audio_loop(tts_response_sentences)
        return
    
    def play_audio_thread(self, audio_data, sample_rate):
        """A separate thread for audio playback."""
        sd.play(audio_data, sample_rate)
        sd.wait()

    def generate_play_audio_loop(self, tts_response_sentences):
        """ a method to generate and play the audio for the chatbot
            args: tts_sentences
            returns: none
        """
        ticker = 0
        last_sentence = None

        for sentence in tts_response_sentences:
            ticker += 1

            if last_sentence is not None and isinstance(last_sentence, str):
                if len(last_sentence) >= len(sentence):
                    sd.wait()

            # Generate TTS audio (replace with your actual TTS logic)
            print("starting speeech generation:")
            tts_audio = self.tts.tts(text=sentence, speaker_wav=self.tts_voice_ref_wav_path, language="en", speed=2.4)

            # Convert to NumPy array (adjust dtype as needed)
            tts_audio = np.array(tts_audio, dtype=np.float32)

            # Create a new WAV file for each sentence
            wav_name_str = f"current_speech_{ticker}.wav"
            wav_paths = {}
            wav_paths[ticker] = f"{self.tts_store_wav_locker_path}\\{wav_name_str}"

            # Write the TTS audio directly to the WAV file
            sf.write(wav_paths[ticker], tts_audio, 22050)

            # Store processed sentence
            last_sentence = sentence
            print(f"Generated WAV file: {wav_paths[ticker]}")

            # Play the audio in a separate thread
            audio_thread = threading.Thread(target=self.play_audio_thread(tts_audio, 22050))
            audio_thread.start()

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
    
    def file_name_voice_filter(self, user_input_agent_name):
        """ a method for preprocessing the voice recognition with a filter before forwarding the agent file names.
            args: user_input_agent_name
            returns: user_input_agent_name
        """
        # Use regex to replace all spaces with underscores
        user_input_agent_name = re.sub(' ', '_', user_input_agent_name)
        return user_input_agent_name