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
import keyboard
import time
from tts_processor_class import tts_processor_class
import speech_recognition as sr
from directory_manager_class import directory_manager_class

class ollama_chatbot_class:
    """ A class for accessing the ollama local serve api via python, and creating new custom agents.
    The ollama_chatbot_class is also used for accessing Speech to Text transcription/Text to Speech Generation methods via a speedy
    low level, command line interface and the Tortoise TTS model.
    """
    def __init__(self, user_input_model_select):
        """ a method for initializing the class
        """
        """ a method for initializing the class
        """
        # User Input
        self.user_input_model_select = user_input_model_select

        # Default Agent Voice Reference
        self.voice_name = "C3PO"
        # Default Movie None
        self.movie_name = None
        self.save_name = "default"
        self.load_name = "default"

        self.url = "http://localhost:11434/api/generate"
        self.headers = {'Content-Type': 'application/json'}
        self.chat_history = []
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.ignored_agents = os.path.join(self.parent_dir, "AgentFiles\\Ignored_Agents\\") 
        self.conversation_library = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\conversation_library")
        self.default_conversation_path = os.path.join(self.parent_dir, f"AgentFiles\\pipeline\\conversation_library\\{self.user_input_model_select}\\{self.save_name}.json")
    
    def send_prompt(self, user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        self.chat_history.append({"user": "User", "message": user_input_prompt})
        # Use only the most recent user input as the prompt
        # prompt = self.chat_history[-1]["message"]

        # join prompt with chat history, 3 turns
        prompt = " ".join([message["message"] for message in self.chat_history[-3:]])

        data = {
            "model": self.user_input_model_select,
            "stream": False,
            "prompt": prompt,
        }

        try:
            # Post the request to the model
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception if the status code is not 200
        except requests.RequestException as e:
            return f"Error: {e}"

        try:
            response_data = json.loads(response.text)
        except json.JSONDecodeError:
            return "Error: Unable to parse response from model"

        if "response" in response_data:
            llama_response = response_data.get("response")
            self.chat_history.append({"user": "Assistant", "message": llama_response})
            return llama_response
        else:
            return "Error: Response from model is not in the expected format"

    def save_to_json(self):
        """ a method for saving the current agent conversation history
            Args: filename
            Returns: none
        """
        file_save_path_dir = os.path.join(self.conversation_library, f"{self.user_input_model_select}")
        file_save_path_str = os.path.join(file_save_path_dir, f"{self.save_name}.json")
        directory_manager_class.create_directory_if_not_exists(file_save_path_dir)
        
        print(f"file path 1:{file_save_path_dir} \n")
        print(f"file path 2:{file_save_path_str} \n")
        with open(file_save_path_str, "w") as json_file:
            json.dump(self.chat_history, json_file, indent=2)

    def load_from_json(self):
        """ a method for loading the directed conversation history to the current agent, mis matching
        agents and history may be bizarre
            Args: filename
            Returns: none
        """
        file_load_path_dir = os.path.join(self.conversation_library, f"{self.user_input_model_select}")
        file_load_path_str = os.path.join(file_load_path_dir, f"{self.load_name}.json")
        directory_manager_class.create_directory_if_not_exists(file_load_path_dir)
        print(f"file path 1:{file_load_path_dir} \n")
        print(f"file path 2:{file_load_path_str} \n")
        with open(file_load_path_str, "r") as json_file:
            self.chat_history = json.load(json_file)

    def create_agent_cmd(self):
        """Executes the create_agent_automation.cmd file with the specified agent name.
            Args: 
            Returns: None
        """
        try:
            # Construct the path to the create_agent_automation.cmd file
            batch_file_path = os.path.join(self.current_dir, "create_agent_automation.cmd")

            # Call the batch file
            subprocess.run(f"call {batch_file_path} {self.user_input_model_select}", shell=True)
        except Exception as e:
            print(f"Error executing create_agent_cmd: {str(e)}")

    def write_model_file_and_run_agent_create(self, listen_flag):
        """ a method to automatically generate a new agent via commands
            args: none
            returns: none
        """
        # collect agent data with stt or ttt
        if listen_flag == False:  # listen_flag is False, use speech recognition
            print("Press space bar to record the new agent's agent name.")
            while not keyboard.is_pressed('space'):  # wait for space bar press
                time.sleep(0.1)
            # Now start recording
            try:
                mic_audio = tts_processor_class.get_audio()
                user_create_agent_name = tts_processor_class.recognize_speech(mic_audio)
                user_create_agent_name = tts_processor_class.file_name_conversation_history_filter(user_create_agent_name)
            except sr.UnknownValueError:
                print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
            except sr.RequestError as e:
                print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

            print("Press space bar to record the new agent's temperature.")
            while not keyboard.is_pressed('space'):  # wait for space bar press
                time.sleep(0.1)
            # Now start recording
            try:
                mic_audio = tts_processor_class.get_audio()
                user_input_temperature = tts_processor_class.recognize_speech(mic_audio)
            except sr.UnknownValueError:
                print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
            except sr.RequestError as e:
                print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

            print("Press space bar to record the new agent's system prompt.")
            while not keyboard.is_pressed('space'):  # wait for space bar press
                time.sleep(0.1)
            # Now start recording
            try:        
                mic_audio = tts_processor_class.get_audio()
                system_prompt = tts_processor_class.recognize_speech(mic_audio)
            except sr.UnknownValueError:
                print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
            except sr.RequestError as e:
                print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

        elif listen_flag == True:  # listen_flag is True, use text input
            user_create_agent_name = input(WARNING + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + OKBLUE)
            user_input_temperature = input(WARNING + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + OKBLUE)
            system_prompt = input(WHITE + "<<< PROVIDE SYSTEM PROMPT >>> " + OKBLUE)

        model_create_dir = os.path.join(self.ignored_agents, f"{user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents, f"{user_create_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)

            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {self.user_input_model_select}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"\n#Set the system prompt\n")
                f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")

            # Execute create_agent_cmd
            self.create_agent_cmd(user_create_agent_name)
            return
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"

    def voice_command_select_filter(self, user_input_prompt):
        """ a method for managing the voice command selection
            Args: user_input_prompt
            Returns: user_input_prompt
        """ 
        user_input_prompt = re.sub(r"activate swap", "/swap", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate quit", "/quit", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate create", "/create", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate listen on", "/listen on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate listen on", "/listen off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate speech on", "/speech on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate speech off", "/speech off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate leap on", "/leap on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate leap off", "/leap off", user_input_prompt, flags=re.IGNORECASE)

        # Search for the name after 'forward slash voice swap'
        match = re.search(r"(activate voice swap|/voice swap) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.voice_name = match.group(2)
            self.voice_name = tts_processor_class.file_name_conversation_history_filter(self.voice_name)

        # Search for the name after 'forward slash movie'
        match = re.search(r"(activate movie|/movie) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.movie_name = match.group(2)
            self.movie_name = tts_processor_class.file_name_conversation_history_filter(self.movie_name)
        else:
            self.movie_name = None

        # Search for the name after 'activate save'
        match = re.search(r"(activate save as|/save as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.save_name = match.group(2)
            self.save_name = tts_processor_class.file_name_conversation_history_filter(self.save_name)
            print(f"save_name string: {self.save_name}")
        else:
            self.save_name = None

        # Search for the name after 'activate load'
        match = re.search(r"(activate load as|/load as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.load_name = match.group(2)
            self.load_name = tts_processor_class.file_name_conversation_history_filter(self.load_name)
            print(f"load_name string: {self.load_name}")
        else:
            self.load_name = None

        return user_input_prompt 
        
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

    # initialize command state flags
    leap_flag = True
    listen_flag = True

    # instantiate class calls
    tts_processor_class = tts_processor_class()
    directory_manager_class = directory_manager_class()

    # begin chatbot loop
    ollama_chatbot_class.user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME >>> " + OKBLUE)
    # new instance class
    ollama_chatbot_class = ollama_chatbot_class(ollama_chatbot_class.user_input_model_select)

    print(OKCYAN + "Press space bar to record audio:" + OKCYAN)
    print(GREEN + f"<<< USER >>> " + END)
    while True:
        user_input_prompt = ""
        speech_done = False
        if listen_flag == True:  # listen_flag is True
            print(OKCYAN + "Please type your selected prompt:" + OKCYAN)
            user_input_prompt = input(GREEN + f"<<< USER >>> " + END)
            speech_done = True
        elif listen_flag == False:
            while keyboard.is_pressed('space'):  # user holds down the space bar
                try:
                    # Record audio from microphone
                    audio = tts_processor_class.get_audio()
                    # Recognize speech to text from audio
                    user_input_prompt = tts_processor_class.recognize_speech(audio)
                    speech_done = True
                except sr.UnknownValueError:
                    print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
                except sr.RequestError as e:
                    print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

        # Use re.sub to replace "forward slash cmd" with "/cmd"
        user_input_prompt = ollama_chatbot_class.voice_command_select_filter(user_input_prompt)

        # if cmd call desired cmd functions
        if user_input_prompt.lower() == "/swap":
            ollama_chatbot_class.chat_history = []
            ollama_chatbot_class.user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME TO SWAP >>> " + OKBLUE)
            print(f"Model changed to {ollama_chatbot_class.user_input_model_select}")

        elif re.match(r"/voice swap ([^/.]*)", user_input_prompt.lower()):
            print(f"Agent voice swapped to {ollama_chatbot_class.voice_name}.json")
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif re.match(r"/save as ([^/.]*)", user_input_prompt.lower()):
            ollama_chatbot_class.save_to_json()
            print(f"Chat history saved to {ollama_chatbot_class.save_name}.json")
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif re.match(r"/load as ([^/.]*)", user_input_prompt.lower()):
            ollama_chatbot_class.load_from_json()
            print(f"Chat history loaded from {ollama_chatbot_class.load_name}.json")
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif user_input_prompt.lower() == "/quit":
            break
        elif user_input_prompt.lower() == "/create":
            ollama_chatbot_class.write_model_file_and_run_agent_create(listen_flag)
            print(GREEN + f"<<< USER >>> " + OKGREEN)
        elif user_input_prompt.lower() == "/listen on":
            listen_flag = True
        elif user_input_prompt.lower() == "/listen off":
            listen_flag = False
            print(GREEN + f"<<< USER >>> " + OKGREEN)
        elif user_input_prompt.lower() == "/leap on":
            leap_flag = True
            print(GREEN + f"<<< USER >>> " + OKGREEN)
        elif user_input_prompt.lower() == "/leap off":
            leap_flag = False
            print(GREEN + f"<<< USER >>> " + OKGREEN)
        elif user_input_prompt.lower() == "/speech on":
            leap_flag = False
            listen_flag = False
            print(GREEN + f"<<< USER >>> " + OKGREEN)
        elif user_input_prompt.lower() == "/speech off":
            leap_flag = True
            listen_flag = True
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif speech_done == True:
            print(YELLOW + f"{user_input_prompt}" + OKCYAN)
            # Send the prompt to the assistant
            response = ollama_chatbot_class.send_prompt(user_input_prompt)
            print(RED + f"<<< {ollama_chatbot_class.user_input_model_select} >>> " + END + f"{response}" + RED)

            # Preprocess for text to speech, add flag for if text to speech enable handle canche otherwise do /leap or smt
            # Clear speech cache and split the response into sentences for next TTS cache
            if leap_flag is not None and isinstance(leap_flag, bool):
                if leap_flag != True:
                    directory_manager_class.clear_directory(tts_processor_class.agent_voice_gen_dump)
                    tts_processor_class.process_tts_responses(response, ollama_chatbot_class.voice_name)
            elif leap_flag is None:
                pass
            print(GREEN + f"<<< USER >>> " + END)

            # DONE Add commands for 0.24: 
            # /voice
             
            #TODO 0.25 
            #/record, /clone voice, /playback, /music play, /movie play

            # TODO 0.26
            # RAG AND GOOGLE API

            # TODO 0.27  
            # TODO LORA AND SORA STABLE DIFFUSION