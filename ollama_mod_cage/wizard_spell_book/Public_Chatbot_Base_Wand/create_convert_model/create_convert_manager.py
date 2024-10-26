""" create_convert_manager.py
"""
import os
import glob
import pyautogui
import time
import subprocess
# from Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands
import shutil
import json

# -------------------------------------------------------------------------------------------------
class create_convert_manager:

    # -------------------------------------------------------------------------------------------------
    def __init__(self, colors, developer_tools_dict):
        """a method for initializing the class
        """
        self.developer_tools_dict = developer_tools_dict
        self.current_dir = self.developer_tools_dict['current_dir']
        self.parent_dir = self.developer_tools_dict['parent_dir']
        self.pipeline = self.developer_tools_dict['ignored_pipeline_dir']
        self.colors = colors

    # -------------------------------------------------------------------------------------------------
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
        print(self.colors['RED'] + f"CONVERTED: {safe_tensor_input_name}" + self.colors['OKGREEN'])
        print(self.colors['GREEN'] + f"<<< USER >>> " + self.colors['OKGREEN'])
        return
    # -------------------------------------------------------------------------------------------------
    def create_agent_cmd(self, user_create_agent_name, cmd_file_name):
        """Executes the create_agent_automation.cmd file with the specified agent name.
            Args: 
            Returns: None
        """
        try:
            # Construct the path to the create_agent_automation.cmd file
            batch_file_path = os.path.join(self.current_dir, cmd_file_name)

            # Call the batch file
            subprocess.run(f"call {batch_file_path} {user_create_agent_name}", shell=True)
        except Exception as e:
            print(f"Error executing create_agent_cmd: {str(e)}")

    # -------------------------------------------------------------------------------------------------
    def copy_gguf_to_ignored_agents(self):
        """ a method to setup the gguf before gguf to ollama create """
        self.create_ollama_model_dir = os.path.join(self.ignored_agents, self.user_create_agent_name)
        print(self.create_ollama_model_dir)
        print(self.gguf_path)
        print(self.create_ollama_model_dir)
        # Copy the file from self.gguf_path to create_ollama_model_dir
        shutil.copy(self.gguf_path, self.create_ollama_model_dir)
        return
    
    # -------------------------------------------------------------------------------------------------
    def write_dict_to_json(self, dictionary, file_path):
        """ a method to write dict to json
        """
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)

        # write_dict_to_json(general_navigator_agent, 'general_navigator_agent.json')

    # -------------------------------------------------------------------------------------------------
    def read_json_to_dict(file_path):

        # # Example usage
        # general_navigator_agent = read_json_to_dict('general_navigator_agent.json')
        # print(general_navigator_agent)


        with open(file_path, 'r') as json_file:
            dictionary = json.load(json_file)
        return dictionary