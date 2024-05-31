""" Ollama public key envrionment variable finder

    This tool created by PromptEngineer48 on github
allows the user to find their ollama key to add to their ollama git account.
This allows the user to push up their own custom models to Ollama.com in the profile.

To use this tool, cd to the file location of key.py in cmd, and run 

"python key.py"

Find out more about PromptEngineer48 at:
        https://github.com/PromptEngineer48/Ollama_custom
"""

import os

# Set the environment variable to the path of id_ed25519.pub
os.environ['PUB_KEY_PATH'] = os.path.expanduser("~/.ollama/id_ed25519.pub")

# Now you can access the environment variable in your Python code
pub_key_path = os.getenv('PUB_KEY_PATH')
print(pub_key_path)  # Verify that the path is correctly set

# Read the contents of the public key file
with open(pub_key_path, 'r') as f:
    public_key = f.read()

# Print the public key
print(public_key)