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

from ollama_chatbot_base import ollama_chatbot_base
from Public_Chatbot_Base_Wand.flags import flag_manager
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
    def __init__(self, instance_name):
        self.instance_name = instance_name  # Add an instance name attribute
        self.user_input_model_select = None  # Add this line before calling super().__init__()
        super().__init__()  # Call the parent class's initializer if necessary
        
    def main(self):
        # Call the main method of the parent class
        super().main()

    @staticmethod
    def run_instance(instance_name):
        chatbot_instance = wizard_chatbot_class(instance_name)
        ollama_chatbot_base.main(chatbot_instance)

# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ 
    The main loop for the ollama_chatbot_class, utilizing a state machine for user command injection during command line prompting,
    all commands start with /, and are named logically.
    """

    # instance_sets = {
    #     "speech": ["main", "gandalf_speech", "merlin_speech"],
    #     # Add more utilities and instance names as needed
    # }

    instance_sets = {
        "speech": ["main_speech"],
        # Add more utilities and instance names as needed
    }

    processes = []
    for utility, instance_names in instance_sets.items():
        for name in instance_names:
            p = multiprocessing.Process(target=wizard_chatbot_class.run_instance, args=(name,))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()