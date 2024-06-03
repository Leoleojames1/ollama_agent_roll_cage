""" driver_script.py

    driver_script.py is the driver for ollama_agent_roll_cage, is a command line interface for STT, 
    & TTS commands with local LLMS. It is an easy to install add on for the ollama application.
    
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

import keyboard
import speech_recognition as sr
import multiprocessing
from multiprocessing import Process, Queue

import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk

from ollama_chatbot_base import ollama_chatbot_base
from Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands
from Public_Chatbot_Base_Wand.speech_to_speech import tts_processor_class
from Public_Chatbot_Base_Wand.directory_manager import directory_manager_class
from Public_Chatbot_Base_Wand.latex_render import latex_render_class
from Public_Chatbot_Base_Wand.data_set_manipulator import data_set_constructor
from Public_Chatbot_Base_Wand.write_modelfile import model_write_class
from Public_Chatbot_Base_Wand.chat_history import json_chat_history
from Public_Chatbot_Base_Wand.read_write_symbol_collector import read_write_symbol_collector

# -------------------------------------------------------------------------------------------------
class wizard_chatbot_class( ollama_chatbot_base ):
    """ a class for setting up the class tool instances and mod tool instances for the defined chatbot instances
    """
    # -------------------------------------------------------------------------------------------------
    def __init__(self, instance_name, user_input_model_select):
        super().__init__(instance_name)
        self.instance_name = instance_name
        self.user_input_model_select = user_input_model_select
        self.colors = self.colors  # Access colors directly
        self.ollama_chatbot_base_instance = None

    # -------------------------------------------------------------------------------------------------
    def get_agent_model_name(self):
        self.user_input_model_select = input(self.colors["HEADER"] + "<<< PROVIDE AGENT NAME >>> " + self.colors["OKBLUE"])
        return self.user_input_model_select
    
    # -------------------------------------------------------------------------------------------------
    def instantiate_ollama_chatbot_base(self):
        self.ollama_chatbot_base_instance = ollama_chatbot_base(self.user_input_model_select) # Instantiate the ollama_chatbot_base class

    # -------------------------------------------------------------------------------------------------
    def start_chatbot_main(self):
        self.instantiate_ollama_chatbot_base()
        self.user_input_model_select = self.user_input_model_select
        self.ollama_chatbot_base_instance.chatbot_main() 
        
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ 
    The main loop for the ollama_chatbot_class, utilizing a state machine for user command injection during command line prompting,
    all commands start with /, and are named logically.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME >>> " + OKBLUE)
    chatbot_instance = wizard_chatbot_class('gandalf', user_input_model_select)
    chatbot_instance.start_chatbot_main()


