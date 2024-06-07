""" json_chat_history.py

    The json_chat_history class provides methods for saving & loading model conversation history, 
    through the use of a custom json library.

created on: May 23, 2024
by @LeoBorcherding
"""

import os
import json

from Public_Chatbot_Base_Wand.directory_manager import directory_manager_class

class json_chat_history:
    """ a class for writing the json history to the json files
    """
    # -------------------------------------------------------------------------------------------------
    def __init__(self, developer_tools_dict):
        """a method for initializing the class
        """
        self.developer_tools_dict = developer_tools_dict
        self.current_dir = developer_tools_dict['current_dir']
        self.parent_dir = developer_tools_dict['parent_dir']
        self.conversation_library = developer_tools_dict['conversation_library_dir']

    # -------------------------------------------------------------------------------------------------
    def save_to_json(self, save_name, user_input_model_select):
        """ a method for saving the current agent conversation history
            Args: filename
            Returns: none
        """
        self.save_name = save_name
        self.user_input_model_select = user_input_model_select
        file_save_path_dir = os.path.join(self.conversation_library, f"{self.user_input_model_select}")
        file_save_path_str = os.path.join(file_save_path_dir, f"{self.save_name}.json")
        directory_manager_class.create_directory_if_not_exists(file_save_path_dir)
        
        print(f"file path 1:{file_save_path_dir} \n")
        print(f"file path 2:{file_save_path_str} \n")
        with open(file_save_path_str, "w") as json_file:
            json.dump(self.chat_history, json_file, indent=2)

    def load_from_json(self, load_name, user_input_model_select):
        """ a method for loading the directed conversation history to the current agent, mis matching
        agents and history may be bizarre
            Args: filename
            Returns: none
        """
        self.load_name = load_name
        self.user_input_model_select = user_input_model_select

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