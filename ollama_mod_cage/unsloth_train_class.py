""" tts_processor.py
    A class for processing the response sentences and audio generation for the ollama_chat_bot_class


    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 Phi-3-mini-4k-instruct
    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 --outfile converted\Phi-3-mini-4k-instruct-q8_0.gguf Phi-3-mini-4k-instruct
"""
import os
import subprocess

class unsloth_train_class:
    def __init__(self):
        """a method for initializing the class
        """
        self.model_git = 'D:\\CodingGit_StorageHDD\\model_git\\'
        self.current_dir = os.getcwd()
    
    def safetensor_ollama_convert(self):
        """ a method for converting safetensor model to ollama model
        """
        #TODO
        return