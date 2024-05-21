""" crew_ai_class_base.py

"""
import os
import subprocess

class crew_ai_class_base:
    def __init__(self):
        """a method for initializing the class
        """
        self.model_git = 'D:\\CodingGit_StorageHDD\\model_git\\'
        self.current_dir = os.getcwd()
        # TODO IF AUTOMATING CMD FILE GENERATION, STORE CMD FILE IN LIBRARY
        self.gen_cmd_lib = "path_here"

    def test_cmd(self, safe_tensor_input_name):
        """ a method for converting safetensors to GGUF
            args: safe_tensor_input_name: str
            returns: None
        """
        # Construct the full path
        full_path = os.path.join(self.current_dir, 'test.cmd')

        # Define the command to be executed
        cmd = f'call {full_path} {self.model_git} {safe_tensor_input_name}'

        # Call the command
        subprocess.run(cmd, shell=True)
        return
    
