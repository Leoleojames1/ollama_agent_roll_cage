""" ollama_commands.py

"""
import os
import ollama
import sys

class ollama_commands:
    def __init__(self, user_input_model_select, developer_tools_dict):
        """a method for initializing the class
        """
        self.user_input_model_select = user_input_model_select
        self.developer_tools_dict = developer_tools_dict
        
        self.current_dir = developer_tools_dict['current_dir']
        self.parent_dir = developer_tools_dict['parent_dir']
    
        self.colors = self.get_colors()

    def get_colors(self):
        """ a method for getting the color dictionary for command line print
            args: none
            returns: self.colors
        """
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
            "WHITE": '\x1B[37m',
            "LIGHT_GREY": '\033[37m',
            "LIGHT_RED": '\033[91m',
            "LIGHT_GREEN": '\033[92m',
            "LIGHT_YELLOW": '\033[93m',
            "LIGHT_BLUE": '\033[94m',
            "LIGHT_MAGENTA": '\033[95m',
            "LIGHT_CYAN": '\033[96m',
            "LIGHT_WHITE": '\033[97m',
            "DARK_BLACK": '\033[30m',
            "DARK_RED": '\033[31m',
            "DARK_GREEN": '\033[32m',
            "DARK_YELLOW": '\033[33m',
            "DARK_BLUE": '\033[34m',
            "DARK_MAGENTA": '\033[35m',
            "DARK_CYAN": '\033[36m',
            "DARK_WHITE": '\033[37m',
            "BRIGHT_BLACK": '\033[90m',
            "BRIGHT_RED": '\033[91m',
            "BRIGHT_GREEN": '\033[92m',
            "BRIGHT_YELLOW": '\033[93m',
            "BRIGHT_BLUE": '\033[94m',
            "BRIGHT_MAGENTA": '\033[95m',
            "BRIGHT_CYAN": '\033[96m',
            "BRIGHT_WHITE": '\033[97m',
        }
        return self.colors
    
    def quit(self):
        """ a method for quitting the program """
        sys.exit()

    def ollama_show_template(self):
        """ a method for getting the model template """
        modelfile_data = ollama.show(f'{self.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key == 'template':
                self.template = value
        return
    
    def ollama_show_license(self):
        """ a method for showing the model license """
        modelfile_data = ollama.show(f'{self.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key == 'license':
                print(self.colors['RED'] + f"<<< {self.user_input_model_select} >>> " + self.colors['OKBLUE'] + f"{key}: {value}")
        return

    def ollama_show_modelfile(self):
        """ a method for showing the modelfile """
        modelfile_data = ollama.show(f'{self.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key != 'license':
                print(self.colors['RED'] + f"<<< {self.user_input_model_select} >>> " + self.colors['OKBLUE'] + f"{key}: {value}")
        return

    def ollama_list(self):
        """ a method for showing the ollama model list """
        ollama_list = ollama.list()
        for model_info in ollama_list.get('models', []):
            model_name = model_info.get('name')
            model = model_info.get('model')
            print(self.colors['RED'] + f"<<< {self.user_input_model_select} >>> " + self.colors['OKBLUE'] + f"{model_name}" + self.colors['RED'] + " <<< ")
        return
    
    def ollama_create(self):
        """ a method for running the ollama create command across the current agent """
        self.model_write_class.write_model_file_and_run_agent_create_ollama(self.listen_flag)
        print(self.colors['GREEN'] + f"<<< USER >>> " + self.colors['OKGREEN'])
        return