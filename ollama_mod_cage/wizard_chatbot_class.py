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

# -------------------------------------------------------------------------------------------------
class wizard_chatbot_class( ollama_chatbot_base ):
    """ a class for setting up the class tool instances and mod tool instances for the defined chatbot instances
    """
    # -------------------------------------------------------------------------------------------------
    def __init__(self, wizard_name):
        """ a method for initializing the wizard_chatbot_class """
        # super().__init__(wizard_name)
        self.wizard_name = wizard_name
        self.ollama_chatbot_base_instance = None

    # -------------------------------------------------------------------------------------------------
    def instantiate_ollama_chatbot_base(self):
        """ a method for Instantiating the ollama_chatbot_base class """
        self.ollama_chatbot_base_instance = ollama_chatbot_base(self.wizard_name) 

    # -------------------------------------------------------------------------------------------------
    def start_chatbot_main(self):
        """ start selected ollama_chatbot_base instance main """
        self.instantiate_ollama_chatbot_base()
        self.ollama_chatbot_base_instance.chatbot_main() 

# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ 
    The main loop for the ollama_chatbot_class, utilizing a state machine for user command injection during command line prompting,
    all commands start with /, and are named logically.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'

    chatbot_instance = wizard_chatbot_class('gandalf')
    chatbot_instance.start_chatbot_main()


