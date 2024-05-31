""" tts_processor.py
    A class for processing the response sentences and audio generation for the ollama_chat_bot_class


    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 Phi-3-mini-4k-instruct
    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 --outfile converted\Phi-3-mini-4k-instruct-q8_0.gguf Phi-3-mini-4k-instruct
"""
import os
import ollama
import sys

class ollama_commands:
    def __init__(self):
        """a method for initializing the class
        """
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        
    @staticmethod
    def get_colors():
        return {
            "YELLOW": '\033[93m',
            "GREEN": '\033[92m',
            "RED": '\033[91m',
            "END": '\033[0m',
            "HEADER": '\033[95m',
            "OKBLUE": '\033[94m',
            "OKCYAN": '\033[96m',
            "OKGREEN": '\033[92m',
            "DARK_GREY": '\033[90m',
            "WARNING": '\033[93m',
            "FAIL": '\033[91m',
            "ENDC": '\033[0m',
            "BOLD": '\033[1m',
            "UNDERLINE": '\033[4m',
            "WHITE": '\x1B[37m'
        }

    def swap(self):
        """ a method to call when swapping models
        """
        self.chat_history = []
        self.user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME TO SWAP >>> " + OKBLUE)
        print(f"Model changed to {self.user_input_model_select}")
        return
    
    def quit(self):
        """ a method for quitting the program """
        sys.exit()

    def ollama_show_template(self, ollama_chatbot_class_instance):
        """ a method for getting the model template """
        modelfile_data = ollama.show(f'{ollama_chatbot_class_instance.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key == 'template':
                ollama_chatbot_class_instance.template = value
        return
    
    def ollama_show_license(self, ollama_chatbot_class_instance):
        """ a method for showing the model license """
        modelfile_data = ollama.show(f'{ollama_chatbot_class_instance.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key == 'license':
                print(self.RED + f"<<< {ollama_chatbot_class_instance.user_input_model_select} >>> " + self.OKBLUE + f"{key}: {value}")
        return

    def ollama_show_modelfile(self, ollama_chatbot_class_instance):
        """ a method for showing the modelfile """
        modelfile_data = ollama.show(f'{ollama_chatbot_class_instance.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key != 'license':
                print(self.RED + f"<<< {ollama_chatbot_class_instance.user_input_model_select} >>> " + self.OKBLUE + f"{key}: {value}")
        return

    def ollama_list(self, ollama_chatbot_class_instance):
        """ a method for showing the ollama model list """
        ollama_list = ollama.list()
        for model_info in ollama_list.get('models', []):
            model_name = model_info.get('name')
            model = model_info.get('model')
            print(self.RED + f"<<< {ollama_chatbot_class_instance.user_input_model_select} >>> " + self.OKBLUE + f"{model_name}" + RED + " <<< ")
        return
    
    def ollama_create(self, ollama_chatbot_class_instance):
        """ a method for running the ollama create command across the current agent """
        ollama_chatbot_class_instance.write_model_file_and_run_agent_create_ollama(ollama_chatbot_class_instance.listen_flag)
        print(self.GREEN + f"<<< USER >>> " + self.OKGREEN)
        return