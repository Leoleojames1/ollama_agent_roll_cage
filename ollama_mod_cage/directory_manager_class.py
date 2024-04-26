"""directory_manager_class.py
    #TODO Finish conversation history, agent .Modelfile, and text to speech voice reference file manager class. 
"""

import os

class directory_manager_class:
    def __init__(self, parent_folder):
        self.parent_folder = parent_folder

    def create_directory_and_text_file(self, file_name):
        """
        Creates a new directory inside the specified parent folder
        and then creates a text file with the given name inside that directory.

        Args:
            file_name (str): Desired name for the text file.

        Returns:
            str: Path to the newly created text file.
        """
        try:
            # Create the new directory
            os.makedirs(self.parent_folder, exist_ok=True)

            # Construct the full path for the text file
            text_file_path = os.path.join(self.parent_folder, file_name)

            # Create the text file
            with open(text_file_path, 'w') as text_file:
                text_file.write("Hello, this is your new text file!")

            return text_file_path
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"

# Example usage:
parent_folder_path = './my_project'
new_file_name = 'my_text_file.txt'

# Create an instance of the DirectoryManager class
dir_manager = directory_manager_class(parent_folder_path)

# Call the method to create the directory and text file
result = dir_manager.create_directory_and_text_file(new_file_name)
print(f"New text file created at: {result}")