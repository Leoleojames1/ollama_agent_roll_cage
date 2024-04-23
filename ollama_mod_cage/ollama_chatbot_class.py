import requests
import json

class ollama_chatbot_class:
    """ A class for accessing the ollama local serve api via python, and creating new custom agents.
    The ollama_chatbot_class is also used for accessing Speech to Text transcription/Text to Speech Generation methods via a speedy
    low level, command line interface and the Tortoise TTS model.
    """
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.headers = {'Content-Type': 'application/json'}
        self.chat_history = []

    def send_prompt(self, user_input_prompt, user_input_model_select):
        """ a method for prompting the current model
            Args: user_input_prompt, user_input_model_select
            Returns: f"Error: {response.status_code} {response.text}"
        """
        self.chat_history.append({"user": "User", "message": user_input_prompt})

        data = {
            "model": f"{user_input_model_select}",
            "stream": False,
            "prompt": "\n".join(msg["message"] for msg in self.chat_history),
        }

        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))

        if response.status_code == 200:
            response_data = json.loads(response.text)
            llama_response = response_data.get("response")
            self.chat_history.append({"user": "Assistant", "message": llama_response})
            return llama_response
        else:
            return f"Error: {response.status_code} {response.text}"

    def save_to_json(self, filename):
        """ a method for saving the current agent conversation history
            Args: filename
            Returns: none
        """
        with open(filename, "w") as json_file:
            json.dump(self.chat_history, json_file, indent=2)

    def load_from_json(self, filename):
        """ a method for loading the directed conversation history to the current agent, mis matching
        agents and history may be bizarre
            Args: filename
            Returns: none
        """
        with open(filename, "r") as json_file:
            self.chat_history = json.load(json_file)

if __name__ == "__main__":
    """ 
    The main loop for the ollama_chatbot_class, utilizing a state machine for user command injection during command line prompting,
    all commands start with /, and are named logically.
    """
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    DARK_GREY = '\033[90m'
    END = '\033[0m'

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    ollama_chatbot_class = ollama_chatbot_class()

    user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME >>> " + OKBLUE)
    while True:
        # Check for user input
        user_input_prompt = input(GREEN + "<<< USER >>> " + OKCYAN)

        # Save the current chat history to chat_history_json
        if user_input_prompt.lower() == "/save":
            ollama_chatbot_class.save_to_json("chat_history.json")
            print("Chat history saved to chat_history.json")

        # Load chat_history.json to the current model
        elif user_input_prompt.lower() == "/load":
            ollama_chatbot_class.load_from_json("chat_history.json")
            print("Chat history loaded from chat_history.json")

        # Clear chat history to allow user to select a new agent
        elif user_input_prompt.lower() == "/swap":
            ollama_chatbot_class.chat_history = []
            user_input_model_select = input(HEADER + "<<< PROVIDE NEW AGENT NAME >>> " + END)
            print(f"Model changed to {user_input_model_select}")

        # Quit back to root directory in cmd
        # elif user_input_prompt.lower() CONTAINS, use REGEX "/search {USER SEARCH REQUEST}":
            #then make google search api request from local machine and provide that data to the model to digest
            #collect /search detection and {USER SEARCH REQUEST} data via regex match groups

        # Quit back to root directory in cmd
        elif user_input_prompt.lower() == "/quit":
            break

        # if not command, then prompt model
        else:
            response = ollama_chatbot_class.send_prompt(user_input_prompt, user_input_model_select)
            print(RED + f"<<< {user_input_model_select} >>> " + END + f"{response}")