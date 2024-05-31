""" flag_manager.py

"""

import os
import time

class flag_manager:
    def __init__(self):
        """a method for initializing the class
        """
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))

    def leap(self, flag, ollama_chatbot_class_instance):
        """ a method for changing the leap flag """
        ollama_chatbot_class_instance.leap_flag = flag
        return
    
    def speech(self, flag, ollama_chatbot_class_instance):
        """ a method for changing the speech flags """
        ollama_chatbot_class_instance.listen_flag = flag
        ollama_chatbot_class_instance.leap_flag = flag
        return
    
    def latex(self, flag, ollama_chatbot_class_instance):
        """ a method for changing the latex flag """
        ollama_chatbot_class_instance.latex_flag = flag
        return
    
    def llava_flow(self, flag, ollama_chatbot_class_instance):
        """ a method for changing the listen flag """
        ollama_chatbot_class_instance.llava_flag = flag
        return
    
    def auto_commands(self, flag, ollama_chatbot_class_instance):
        """ a method for auto_command flag """
        ollama_chatbot_class_instance.auto_commands_flag = flag
        return
    
    def listen(self, flag, ollama_chatbot_class_instance):
        """ a method for changing the listen flag """
        ollama_chatbot_class_instance.listen_flag = flag
        print(f"SET AUTO SPEECH FLAG STATE: {self.listen_flag}")
        return
    
    def chunk_speech(self, value, ollama_chatbot_class_instance):
        time.sleep(1)
        ollama_chatbot_class_instance.chunk_flag = value
        print(f"CHUNK FLAG STATE: {self.chunk_flag}")

    def auto_speech_set(self, value, ollama_chatbot_class_instance):
        ollama_chatbot_class_instance.auto_speech_flag = value
        print(f"AUTO FLAG STATE: {self.auto_speech_flag}")