""" ollama_chatbot_class.py

    ollama_agent_roll_cage, is a command line interface for STT, & TTS commands with local LLMS.
    It is an easy to install add on for the ollama application.
    
        This software was designed by Leo Borcherding with the intent of creating an easy to use
    ai interface for anyone, through Speech to Text and Text to Speech.
        
        With ollama_agent_roll_cage we can provide hands free access to LLM data. 
    This has a host of applications and I want to bring this software to users 
    suffering from blindness/vision loss, and children suffering from austism spectrum 
    disorder as way for learning and expanding communication and speech. 
    
        The C3PO ai is a great imaginary friend! I could envision myself 
    talking to him all day telling me stories about a land far far away! 
    This makes learning fun and accessible! Children would be directly 
    rewarded for better speech as the ai responds to subtle differences 
    in language ultimately educating them without them realizing it.

    Development for this software was started on: 4/20/2024 
    By: Leo Borcherding
        on github @
            leoleojames1/ollama_agent_roll_cage

"""
import os
import subprocess
import requests
import json
import re
import speech_recognition as sr
import keyboard
from tts_processor import tts_processor

class ollama_chatbot_class:
    """ A class for accessing the ollama local serve api via python, and creating new custom agents.
    The ollama_chatbot_class is also used for accessing Speech to Text transcription/Text to Speech Generation methods via a speedy
    low level, command line interface and the Tortoise TTS model.
    """
    def __init__(self):
        """ a method for initializing the class
        """
        self.url = "http://localhost:11434/api/generate"
        self.headers = {'Content-Type': 'application/json'}
        self.chat_history = []
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.ignored_agents = os.path.join(self.parent_dir, "AgentFiles\\Ignored_Agents\\") 
    
    def get_audio(self):
        """ a method for collecting the audio from the microphone
            args: none
            returns: audio
        """
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
        return audio
    
    def recognize_speech(self, audio):
        """ a method for calling the speech recognizer
        """
        return sr.Recognizer().recognize_google(audio)
    
    def send_prompt(self, user_input_prompt, user_input_model_select):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select
            returns: none
        """
        self.chat_history.append({"user": "User", "message": user_input_prompt})

        # Use only the most recent user input as the prompt
        prompt = self.chat_history[-1]["message"]

        data = {
            "model": user_input_model_select,
            "stream": False,
            "prompt": prompt,
        }

        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception if the status code is not 200
            response_data = json.loads(response.text)
            llama_response = response_data.get("response")
            self.chat_history.append({"user": "Assistant", "message": llama_response})
            return llama_response
        except requests.RequestException as e:
            return f"Error: {e}"

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

    def write_model_file_call_and_agent_automation(self):
        """ a method to automatically generate a new agent via commands
            args: none
            returns: none
        """
        # collect agent data
        user_input_agent_name = input(WHITE + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + OKBLUE)
        user_input_temperature = input(WHITE + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + OKBLUE)
        # print("Press space bar to record the new agent's system prompt.")
        # mic_audio = self.get_audio()
        # system_prompt = self.recognize_speech(mic_audio)
        system_prompt = input(WHITE + "<<< PROVIDE SYSTEM PROMPT >>> " + OKBLUE)
        model_create_dir = os.path.join(self.ignored_agents, f"{user_input_agent_name}")
        model_create_file = os.path.join(self.ignored_agents, f"{user_input_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)

            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {user_input_model_select}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"\n#Set the system prompt\n")
                f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")

            # Execute create_agent_cmd
            self.create_agent_cmd(user_input_agent_name)
            return
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"
        
    def create_agent_cmd(self, user_input_agent_name):
        """Executes the create_agent_automation.cmd file with the specified agent name.
            Args: agent_name (str): The name of the agent.
            Returns: None
        """
        try:
            # Construct the path to the create_agent_automation.cmd file
            batch_file_path = os.path.join(self.current_dir, "create_agent_automation.cmd")

            # Call the batch file
            subprocess.run(f"call {batch_file_path} {user_input_agent_name}", shell=True)
        except Exception as e:
            print(f"Error executing create_agent_cmd: {str(e)}")

if __name__ == "__main__":
    """ 
    The main loop for the ollama_chatbot_class, utilizing a state machine for user command injection during command line prompting,
    all commands start with /, and are named logically.
    """
    # Used Colors
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'

    # Unused Colors
    DARK_GREY = '\033[90m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\x1B[37m'

    # instantiate class calls
    ollama_chatbot_class = ollama_chatbot_class()
    tts_processor = tts_processor()

    # begin chatbot loop
    user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME >>> " + OKBLUE)
    print("Press space bar to record audio.")
    print(GREEN + f"<<< USER >>> " + END)
    while True:
        user_input_prompt = ""
        # user_input_prompt = input()
        speech_done = False
        while keyboard.is_pressed('space'):  # user holds down the space bar
            audio = ollama_chatbot_class.get_audio() # record audio
            try:
                # Recognize speech
                user_input_prompt = ollama_chatbot_class.recognize_speech(audio)

                # Use re.sub to replace "forward slash cmd" with "/cmd"
                user_input_prompt = re.sub(r"forward slash swap", "/swap", user_input_prompt, flags=re.IGNORECASE)
                user_input_prompt = re.sub(r"forward slash save", "/save", user_input_prompt, flags=re.IGNORECASE)
                user_input_prompt = re.sub(r"forward slash load", "/load", user_input_prompt, flags=re.IGNORECASE)
                user_input_prompt = re.sub(r"forward slash quit", "/quit", user_input_prompt, flags=re.IGNORECASE)
                user_input_prompt = re.sub(r"forward slash create", "/create", user_input_prompt, flags=re.IGNORECASE)

                # TODO Add commands for 0.21: 
                # /create, /save as, /load as, 
                
                # TODO Add commands for 0.22: 
                # /speech, /listen, /leep

                # TODO Add commands for 0.23: 
                # /voice, /record, /clone voice, /playback, /music play, /movie play

                # if cmd call desired cmd functions
                if user_input_prompt.lower() == "/swap":
                    ollama_chatbot_class.chat_history = []
                    user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME TO SWAP >>> " + OKBLUE)
                    print(f"Model changed to {user_input_model_select}")
                elif user_input_prompt.lower() == "/save":
                    ollama_chatbot_class.save_to_json("chat_history.json")
                    print("Chat history saved to chat_history.json")
                elif user_input_prompt.lower() == "/load":
                    ollama_chatbot_class.load_from_json("chat_history.json")
                    print("Chat history loaded from chat_history.json")
                elif user_input_prompt.lower() == "/quit":
                    break
                elif user_input_prompt.lower() == "/create":
                    ollama_chatbot_class.write_model_file_call_and_agent_automation()
                # Speech done
                speech_done = True
            # Flip speech recognition errors
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

                # If speech recognition was successful, proceed with the rest of the loop
                if speech_done:
                    print(YELLOW + f"{user_input_prompt}" + OKCYAN)
                    # Send the prompt to the assistant
                    response = ollama_chatbot_class.send_prompt(user_input_prompt, user_input_model_select)
                    print(RED + f"<<< {user_input_model_select} >>> " + END + f"{response}" + RED)

                    # Split the response into sentences for TTS Multiprocessing
                    tts_processor.process_tts_responses(response)
                    print(GREEN + f"<<< USER >>> " + END)