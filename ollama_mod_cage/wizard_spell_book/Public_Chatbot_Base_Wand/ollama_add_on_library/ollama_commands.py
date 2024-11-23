""" ollama_commands.py

"""
import os
import ollama
import sys
import logging

class ollama_commands:
    def __init__(self, user_input_model_select, pathLibrary):
        """a method for initializing the class
        """
        self.user_input_model_select = user_input_model_select
        self.pathLibrary = pathLibrary
        
        self.current_dir = pathLibrary['current_dir']
        self.parent_dir = pathLibrary['parent_dir']
    
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

    async def ollama_show_loaded_models(self):
        ollama_loaded_models = ollama.ps()
        return ollama_loaded_models

    async def ollama_show_template(self):
        modelfile_data = ollama.show(f'{self.user_input_model_select}')
        return modelfile_data.get('template', '')
    
    async def ollama_show_license(self):
        modelfile_data = ollama.show(f'{self.user_input_model_select}')
        return modelfile_data.get('license', '')

    async def ollama_show_modelfile(self):
        return ollama.show(f'{self.user_input_model_select}')

    async def ollama_list(self):
        """Get list of available models"""
        try:
            result = ollama.list()
            if not isinstance(result, dict) or 'models' not in result:
                logging.warning("Unexpected response format from ollama.list()")
                return []
                
            # Extract and format model names
            models = [model['name'] for model in result['models'] if 'name' in model]
            return models
            
        except Exception as e:
            logging.error(f"Error listing models: {str(e)}")
            return []

    async def ollama_create(self):
        # Implement this method to create a new Ollama model
        # Return the result instead of printing it
        #TODO IMPLEMENT
        pass
    
    # def ollama_create(self):
    #     """ a method for running the ollama create command across the current agent """
    #     self.model_write_class.write_model_file_and_run_agent_create_ollama(self.listen_flag)
    #     print(self.colors['GREEN'] + f"<<< USER >>> " + self.colors['OKGREEN'])
    #     return