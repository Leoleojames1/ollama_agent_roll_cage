""" ollama_commands.py

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
        self.parent_dir = os.path.abspath(os.path.join(self.parent_dir, os.pardir))
        
        self.colors = self.get_colors()

    def get_colors(self):
        self.colors = {
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
        return self.colors
    
    def swap(self):
        """ a method to call when swapping models
        """
        self.chat_history = []
        self.user_input_model_select = input(self.colors['HEADER']+ "<<< PROVIDE AGENT NAME TO SWAP >>> " + self.colors['OKBLUE'])
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
                print(self.colors['RED'] + f"<<< {ollama_chatbot_class_instance.user_input_model_select} >>> " + self.colors['OKBLUE'] + f"{key}: {value}")
        return

    def ollama_show_modelfile(self, ollama_chatbot_class_instance):
        """ a method for showing the modelfile """
        modelfile_data = ollama.show(f'{ollama_chatbot_class_instance.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key != 'license':
                print(self.colors['RED'] + f"<<< {ollama_chatbot_class_instance.user_input_model_select} >>> " + self.colors['OKBLUE'] + f"{key}: {value}")
        return

    def ollama_list(self, ollama_chatbot_class_instance):
        """ a method for showing the ollama model list """
        ollama_list = ollama.list()
        for model_info in ollama_list.get('models', []):
            model_name = model_info.get('name')
            model = model_info.get('model')
            print(self.colors['RED'] + f"<<< {ollama_chatbot_class_instance.user_input_model_select} >>> " + self.colors['OKBLUE'] + f"{model_name}" + self.colors['RED'] + " <<< ")
        return
    
    def ollama_create(self, ollama_chatbot_class_instance):
        """ a method for running the ollama create command across the current agent """
        ollama_chatbot_class_instance.write_model_file_and_run_agent_create_ollama(ollama_chatbot_class_instance.listen_flag)
        print(self.colors['GREEN'] + f"<<< USER >>> " + self.colors['OKGREEN'])
        return