""" tts_processor.py

    A class for processing the response sentences and audio generation for the ollama_chat_bot_class
"""

import sounddevice as sd
import os
import torch
import re
from TTS.api import TTS

class tts_processor:
    def __init__(self):
        """a method for initializing the class
        """
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_wav_path = os.path.join(self.parent_dir, "AgentFiles\\Ignored_TTS\\pipeline\\active_group\\clone_speech.wav")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def process_tts_responses(self, response):
        """a method for managing the response preprocessing methods
            args: response
            returns: none
        """
        # Call Sentence Splitter
        tts_response_sentences = self.split_into_sentences(response)
        self.generate_play_audio_loop(tts_response_sentences)
        return

    def generate_play_audio_loop(self, tts_response_sentences):
        """a method for generating and playing the chatbot audio loops
            args: tts_response_sentences
            none: none
        """
        # Generate audio for each sentence in TTS
        flag = True
        for sentence in tts_response_sentences:
            # If not first wave wait again
            if flag == False:
                sd.wait()
            # Generate wav file
            tts_audio = self.tts.tts(text=sentence, speaker_wav=(f"{self.tts_wav_path}"), language="en")
            # Play Audio
            sd.play(tts_audio, samplerate=22050)
            # If not first wav file wait
            if flag == False:
                sd.wait()
            flag = False
        return

    def split_into_sentences(self, text: str) -> list[str]:
        """a method for splitting the llm response into sentences
            args: text: str -> list[str]
            returns: sentences
        """
        # Add spaces around punctuation marks for consistent splitting
        text = " " + text + " "
        text = text.replace("\n", " ")

        # Handle common abbreviations and special cases
        text = re.sub(r"(Mr|Mrs|Ms|Dr|i\.e)\.", r"\1<prd>", text)
        text = re.sub(r"\.\.\.", r"<prd><prd><prd>", text)

        # Split on period, question mark, or exclamation mark followed by optional spaces
        sentences = re.split(r"[.!?]\s*", text)

        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences