"""directory_manager_class.py
    #TODO Finish conversation history, agent .Modelfile, and text to speech voice reference file manager class. 
"""

import os
import shutil

class directory_manager_class:
    def __init__(self):
        self.test = "test"

    def clear_directory(self, directory_path):
        try:
            # Remove all files in the directory
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Remove all subdirectories (non-empty) in the directory
            for subdirectory in os.listdir(directory_path):
                subdirectory_path = os.path.join(directory_path, subdirectory)
                if os.path.isdir(subdirectory_path):
                    shutil.rmtree(subdirectory_path)

            print(f"Successfully cleared everything in {directory_path}")
        except Exception as e:
            print(f"Error while clearing directory: {e}")

    def create_directory_if_not_exists(self, directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_path}' already exists.")