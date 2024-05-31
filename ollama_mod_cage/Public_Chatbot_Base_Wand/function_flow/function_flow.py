""" function_flow.py
"""
import os
import glob
import pyautogui
import time
import subprocess

class function_flow:
    def __init__(self):
        """a method for initializing the class
        """
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.pipeline = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\")
        #TODO GET function_prompt.txt, command list data and prompt model
        
    def command_auto_select_list_generate(self):
        """ a method to generate a command list to be executed by the function call model
        """
        return

    def command_function_call_model(self):
        """ a method to call the command list produced by command_auto_select_list_generate
        """
        return