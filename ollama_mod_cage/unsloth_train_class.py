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


    def safe_tensor_gguf_convert(self, safe_tensor_input_name):
        """ a method for converting safetensors to GGUF
            args: safe_tensor_input_name: str
            returns: None
        """
        # Construct the full path
        full_path = os.path.join(self.current_dir, 'safetensors_to_GGUF.cmd')

        # Define the command to be executed
        cmd = f'call {full_path} {self.model_git} {safe_tensor_input_name}'

        # Call the command
        subprocess.run(cmd, shell=True)
        return
    
    def safetensor_ollama_convert(self):
        """ a method for converting safetensor model to ollama model
        """
        #TODO
        return