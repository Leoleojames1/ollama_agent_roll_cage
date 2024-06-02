""" create_convert_manager.py
"""
import os
import glob
import pyautogui
import time
import subprocess
import json

from Public_Chatbot_Base_Wand.directory_manager import directory_manager_class

class json_chat_history:
    def __init__(self):
        """a method for initializing the class
        """
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.parent_dir = os.path.abspath(os.path.join(self.parent_dir, os.pardir))
        self.pipeline = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\")

    def save_to_json(self):
        """ a method for saving the current agent conversation history
            Args: filename
            Returns: none
        """
        file_save_path_dir = os.path.join(self.conversation_library, f"{self.user_input_model_select}")
        file_save_path_str = os.path.join(file_save_path_dir, f"{self.save_name}.json")
        directory_manager_class.create_directory_if_not_exists(file_save_path_dir)
        
        print(f"file path 1:{file_save_path_dir} \n")
        print(f"file path 2:{file_save_path_str} \n")
        with open(file_save_path_str, "w") as json_file:
            json.dump(self.chat_history, json_file, indent=2)

    def load_from_json(self):
        """ a method for loading the directed conversation history to the current agent, mis matching
        agents and history may be bizarre
            Args: filename
            Returns: none
        """
        # Check if user_input_model_select contains a slash
        if "/" in self.user_input_model_select:
            user_dir, model_dir = self.user_input_model_select.split("/")
            file_load_path_dir = os.path.join(self.conversation_library, user_dir, model_dir)
        else:
            file_load_path_dir = os.path.join(self.conversation_library, self.user_input_model_select)

        file_load_path_str = os.path.join(file_load_path_dir, f"{self.load_name}.json")
        directory_manager_class.create_directory_if_not_exists(file_load_path_dir)
        print(f"file path 1:{file_load_path_dir} \n")
        print(f"file path 2:{file_load_path_str} \n")
        with open(file_load_path_str, "r") as json_file:
            self.chat_history = json.load(json_file)