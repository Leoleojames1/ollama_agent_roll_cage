""" ollama_chatbot_class.py

    ollama_agent_roll_cage, is a command line interface for STT, & TTS commands with local LLMS.
    It is an easy to install add on for the ollama application.
    
        This software was designed by Leo Borcherding with the intent of creating an easy to use
    ai interface for anyone, through Speech to Text and Text to Speech.
        
        With ollama_agent_roll_cage we can provide hands free access to LLM data. 
    This has a host of applications and I want to bring this software to users 
    suffering from blindness/vision loss, and children suffering from austism spectrum 
    disorder as way for learning and expanding communication and speech. 
    
        The C3PO ai is a great imaginary friend! I could envision myself 
    talking to him all day telling me stories about a land far far away! 
    This makes learning fun and accessible! Children would be directly 
    rewarded for better speech as the ai responds to subtle differences 
    in language ultimately educating them without them realizing it.

    Development for this software was started on: 4/20/2024 
    By: Leo Borcherding
        on github @ 
            leoleojames1/ollama_agent_roll_cage

"""
import os
import sys
import subprocess
import json
import re
import keyboard
import time
import speech_recognition as sr
import ollama
import shutil
import threading
import pyautogui
import glob
from PIL import Image
import base64

from Public_Chatbot_Base_Wand.flags import flag_manager
from Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands
from Public_Chatbot_Base_Wand.speech_to_speech import tts_processor_class
from Public_Chatbot_Base_Wand.directory_manager import directory_manager_class
from Public_Chatbot_Base_Wand.latex_render import latex_render_class
from Public_Chatbot_Base_Wand.data_set_manipulator import data_set_constructor
from Public_Chatbot_Base_Wand.write_modelfile import model_write_class
from Public_Chatbot_Base_Wand.chat_history import json_chat_history
from Public_Chatbot_Base_Wand.read_write_symbol_collector import read_write_symbol_collector

# from tensorflow.keras.models import load_model
# sentiment_model = load_model('D:\\CodingGit_StorageHDD\\model_git\\emotions_classifier\\emotions_classifier.keras')

# -------------------------------------------------------------------------------------------------
class ollama_chatbot_class:
    """ A class for accessing the ollama local serve api via python, and creating new custom agents.
    The ollama_chatbot_class is also used for accessing Speech to Text transcription/Text to Speech Generation methods via a speedy
    low level, command line interface and the Tortoise TTS model.
    """

    # -------------------------------------------------------------------------------------------------
    def __init__(self, ):
        """ a method for initializing the class """
        # Connect api
        self.url = "http://localhost:11434/api/chat"
        # Setup chat_history
        self.headers = {'Content-Type': 'application/json'}
        self.chat_history = []
        self.llava_history = []
        # Default Agent Voice Reference
        self.voice_name = "C3PO"
        # Default conversation name
        self.save_name = "default"
        self.load_name = "default"

        #TODO ADD FILE PATH COLLECTOR, MANAGER, PARSER & a developer_tools.txt to house said paths.
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.ignored_agents = os.path.join(self.parent_dir, "AgentFiles\\Ignored_Agents\\") 
        self.conversation_library = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\conversation_library")
        self.default_conversation_path = os.path.join(self.parent_dir, f"AgentFiles\\pipeline\\conversation_library\\{self.user_input_model_select}\\{self.save_name}.json")
        self.llava_library = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\llava_library")

        # TODO developer_tools.txt file for custom path library
        self.model_git = 'D:\\CodingGit_StorageHDD\\model_git\\'

        #Initialize tool flags
        self.leap_flag = True # TODO TURN OFF FOR MINECRAFT
        self.listen_flag = False # TODO TURN ON FOR MINECRAFT
        self.latex_flag = False
        self.llava_flag = False # TODO TURN ON FOR MINECRAFT
        self.chunk_flag = False
        self.auto_speech_flag = False #TODO KEEP OFF BY DEFAULT FOR MINECRAFT, TURN ON TO START
        self.splice_flag = False

    # -------------------------------------------------------------------------------------------------
    def select_model(self, user_input_model_select):
        self.user_input_model_select = user_input_model_select
        return user_input_model_select

    # -------------------------------------------------------------------------------------------------
    def voice_command_select_filter(self, user_input_prompt):
        """ a method for managing the voice command selection
            Args: user_input_prompt
            Returns: user_input_prompt
        """ 
        user_input_prompt = re.sub(r"activate swap", "/swap", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate quit", "/quit", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate create", "/create", user_input_prompt, flags=re.IGNORECASE)

        user_input_prompt = re.sub(r"activate listen on", "/listen on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate listen on", "/listen off", user_input_prompt, flags=re.IGNORECASE)

        user_input_prompt = re.sub(r"activate speech on", "/speech on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate speech off", "/speech off", user_input_prompt, flags=re.IGNORECASE)

        user_input_prompt = re.sub(r"activate leap on", "/leap on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate leap off", "/leap off", user_input_prompt, flags=re.IGNORECASE)

        user_input_prompt = re.sub(r"activate latex on", "/latex on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate latex off", "/latex off", user_input_prompt, flags=re.IGNORECASE)

        user_input_prompt = re.sub(r"activate show model", "/show model", user_input_prompt, flags=re.IGNORECASE)

        # Search for the name after 'forward slash voice swap'
        match = re.search(r"(activate voice swap|/voice swap) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.voice_name = match.group(2)
            self.voice_name = tts_processor_instance.file_name_conversation_history_filter(self.voice_name)

        # Search for the name after 'forward slash movie'
        match = re.search(r"(activate movie|/movie) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.movie_name = match.group(2)
            self.movie_name = tts_processor_instance.file_name_conversation_history_filter(self.movie_name)
        else:
            self.movie_name = None

        # Search for the name after 'activate save'
        match = re.search(r"(activate save as|/save as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.save_name = match.group(2)
            self.save_name = tts_processor_instance.file_name_conversation_history_filter(self.save_name)
            print(f"save_name string: {self.save_name}")
        else:
            self.save_name = None

        # Search for the name after 'activate load'
        match = re.search(r"(activate load as|/load as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.load_name = match.group(2)
            self.load_name = tts_processor_instance.file_name_conversation_history_filter(self.load_name)
            print(f"load_name string: {self.load_name}")
        else:
            self.load_name = None

        # Search for the name after 'forward slash voice swap'
        match = re.search(r"(activate convert tensor|/convert tensor) ([^\s]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.tensor_name = match.group(2)

        return user_input_prompt 
    
    # -------------------------------------------------------------------------------------------------
    def command_select(self, command_str):
        """ a method for selecting the command to execute
            Args: command_str
            Returns: command_library[command_str]
        """
        command_library = {
            "/swap": lambda: ollama_command_instance.swap(),
            "/voice swap": lambda: tts_processor_instance.voice_swap(),
            "/save as": lambda: json_chat_history_instance.save_to_json(),
            "/load as": lambda: json_chat_history_instance.load_from_json(),
            "/write modelfile": lambda: model_write_class_instance.write_model_file(),
            "/convert tensor": lambda: model_write_class_instance.safe_tensor_gguf_convert(self.tensor_name),
            "/convert gguf": lambda: model_write_class_instance.write_model_file_and_run_agent_create_gguf(self.listen_flag, self.model_git),
            "/listen on": lambda: flag_manager_instance.listen(True, ollama_chatbot_class_instance),
            "/listen off": lambda: flag_manager_instance.listen(False, ollama_chatbot_class_instance),
            "/leap on": lambda: flag_manager_instance.leap(True, ollama_chatbot_class_instance),
            "/leap off": lambda: flag_manager_instance.leap(False, ollama_chatbot_class_instance),
            "/speech on": lambda: flag_manager_instance.speech(True, ollama_chatbot_class_instance),
            "/speech off": lambda: flag_manager_instance.speech(False, ollama_chatbot_class_instance),
            "/latex on": lambda: flag_manager_instance.latex(True, ollama_chatbot_class_instance),
            "/latex off": lambda: flag_manager_instance.latex(False, ollama_chatbot_class_instance),
            "/command auto on": lambda: flag_manager_instance.auto_commands(True, ollama_chatbot_class_instance),
            "/command auto off": lambda: flag_manager_instance.auto_commands(False, ollama_chatbot_class_instance),
            "/llava flow": lambda: flag_manager_instance.llava_flow(True),
            "/llava freeze": lambda: flag_manager_instance.llava_flow(False),
            "/auto on": lambda: flag_manager_instance.auto_speech_set(True),
            "/auto off": lambda: flag_manager_instance.auto_speech_set(False),
            "/quit": lambda: ollama_command_instance.quit(),
            "/ollama create": lambda: ollama_command_instance.ollama_create(ollama_chatbot_class_instance),
            "/ollama show": lambda: ollama_command_instance.ollama_show_modelfile(ollama_chatbot_class_instance),
            "/ollama template": lambda: ollama_command_instance.ollama_show_template(self, ollama_chatbot_class_instance),
            "/ollama license": lambda: ollama_command_instance.ollama_show_license(ollama_chatbot_class_instance),
            "/ollama list": lambda: ollama_command_instance.ollama_list(ollama_chatbot_class_instance),
            "/splice video": lambda: data_set_video_process_instance.generate_image_data(),
            "/developer new" : lambda: read_write_symbol_collector_instance.developer_tools_generate()
        }

        # Find the command in the command string
        command = next((cmd for cmd in command_library.keys() if command_str.startswith(cmd)), None)

        # If a command is found, split it from the arguments
        if command:
            args = command_str[len(command):].strip()
        else:
            args = None

        # Check if the command is in the library, if not return None
        if command in command_library:
            command_library[command]()
            cmd_run_flag = True
            return cmd_run_flag
        else:
            cmd_run_flag = False
            return cmd_run_flag