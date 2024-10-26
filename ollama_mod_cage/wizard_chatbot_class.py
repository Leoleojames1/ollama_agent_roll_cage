""" wizard_chatbot_class.py

    wizard_chatbot_class.py is the driver for ollama_agent_roll_cage, offering a place to define chatbot
    instances, and their toolsets, with the goal of automating an Agent workflow build list defined for 
    each instance. 
        
        This software was designed by Leo Borcherding with the intent of creating an easy to use
    ai interface for anyone, through Speech to Text and Text to Speech.

    Development for this software was started on: 4/20/2024 
    By: Leo Borcherding
        on github @ 
            leoleojames1/ollama_agent_roll_cage

"""

from ollama_chatbot_base import ollama_chatbot_base
import curses
import threading
import time

# -------------------------------------------------------------------------------------------------
class wizard_chatbot_class:
    """ 
    This class sets up the instances of chatbots and manages their interactions.
    """
    # -------------------------------------------------------------------------------------------------
    def __init__(self):
        """ 
        Initialize the wizard_chatbot_class with an empty list of chatbots, 
        a current_chatbot_index set to 0, and a threading lock.
        """
        self.chatbot = None
        self.current_chatbot_index = 0  # Initialize current_chatbot_index
        self.lock = threading.Lock()  # Create a lock

    # -------------------------------------------------------------------------------------------------
    def instantiate_ollama_chatbot_base(self):
        """ a method for Instantiating the ollama_chatbot_base class """
        self.ollama_chatbot_base_instance = ollama_chatbot_base() 

    # -------------------------------------------------------------------------------------------------
    def start_chatbot_main(self):
        """ start selected ollama_chatbot_base instance main """
        self.instantiate_ollama_chatbot_base()
        self.ollama_chatbot_base_instance.chatbot_main() 

# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ 
    The main loop for the ollama_chatbot_class. It uses a state machine for user command injection during command line prompting.
    All commands start with /, and are named logically.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'

    chatbot_instance = wizard_chatbot_class()
    chatbot_instance.start_chatbot_main()
    # curses.wrapper(chatbot_instance.start_chatbot_main)