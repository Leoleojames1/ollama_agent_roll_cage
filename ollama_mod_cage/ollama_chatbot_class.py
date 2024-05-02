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
    def __init__(self):
        """ a method for initializing the class
        """
        self.url = "http://localhost:11434/api/generate"
        self.headers = {'Content-Type': 'application/json'}
        self.chat_history = []
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.ignored_agents = os.path.join(self.parent_dir, "AgentFiles\\Ignored_Agents\\") 
    
    def send_prompt(self, user_input_prompt, user_input_model_select, search_google=False):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        self.chat_history.append({"user": "User", "message": user_input_prompt})

        # Use only the most recent user input as the prompt
        prompt = self.chat_history[-1]["message"]

        # If search_google is True, perform a Google search and use the results as input to the model
        if search_google == True:
            google_search_data, google_image_search_data = self.google_search(prompt)
            # Use the Google search results as the prompt
            prompt = google_search_data  # Replace with the actual data you want to use from the Google search results

        data = {
            "model": user_input_model_select,
            "stream": False,
            "prompt": prompt,
        }

        try:
            # Post the request to the model
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception if the status code is not 200
            response_data = json.loads(response.text)
            llama_response = response_data.get("response")
            self.chat_history.append({"user": "Assistant", "message": llama_response})

            return llama_response
        except requests.RequestException as e:
            return f"Error: {e}"

    def google_search(self, prompt):
        """ a method for Google Search and Google Image Search
            args: prompt
            returns: search results
        """
        # Now let's integrate Google Search API
        google_search_url = "https://www.googleapis.com/customsearch/v1"  # Replace with the actual Google Search API endpoint
        google_search_params = {
            "key": "Your-Google-Search-API-Key",
            "cx": "Your-Custom-Search-ID",
            "q": prompt,
        }
        google_search_response = requests.get(google_search_url, params=google_search_params)
        google_search_data = google_search_response.json()

        # Similarly, integrate Google Image Search API
        google_image_search_url = "https://www.googleapis.com/customsearch/v1"  # Replace with the actual Google Image Search API endpoint
        google_image_search_params = {
            "key": "Your-Google-Image-Search-API-Key",
            "cx": "Your-Custom-Search-ID",
            "q": prompt,
            "searchType": "image",
        }
        google_image_search_response = requests.get(google_image_search_url, params=google_image_search_params)
        google_image_search_data = google_image_search_response.json()

        # Process the responses from Google APIs and integrate them into your chatbot's response mechanism
        # ...

        return google_search_data, google_image_search_data

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
                user_input_agent_name = tts_processor_class.recognize_speech(mic_audio)
                user_input_agent_name = tts_processor_class.file_name_voice_filter(user_input_agent_name)
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
            user_input_agent_name = input(WARNING + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + OKBLUE)
            user_input_temperature = input(WARNING + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + OKBLUE)
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

    def voice_command_select_filter(self, user_input_prompt):
        """ a method for managing the voice command selection
            Args: user_input_prompt
            Returns: user_input_prompt
        """ 
        user_input_prompt = re.sub(r"forward slash swap", "/swap", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash save", "/save", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash load", "/load", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash quit", "/quit", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash create", "/create", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash listen on", "/listen on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash listen on", "/listen off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash speech on", "/speech on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash speech off", "/speech off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash leap on", "/leap on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"forward slash leap off", "/leap off", user_input_prompt, flags=re.IGNORECASE)

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

    # instantiate class calls
    ollama_chatbot_class = ollama_chatbot_class()
    tts_processor_class = tts_processor_class()
    directory_manager_class = directory_manager_class()

    # initialize command state flags
    leap_flag = True
    listen_flag = True
    
    # begin chatbot loop
    user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME >>> " + OKBLUE)
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
            user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME TO SWAP >>> " + OKBLUE)
            print(f"Model changed to {user_input_model_select}")

        elif user_input_prompt.lower() == "/save":
            ollama_chatbot_class.save_to_json("chat_history.json")
            print("Chat history saved to chat_history.json")
            print(GREEN + f"<<< USER >>> " + END)
        elif user_input_prompt.lower() == "/load":
            ollama_chatbot_class.load_from_json("chat_history.json")
            print("Chat history loaded from chat_history.json")
            print(GREEN + f"<<< USER >>> " + END)
        elif user_input_prompt.lower() == "/quit":
            break
        elif user_input_prompt.lower() == "/create":
            ollama_chatbot_class.write_model_file_and_run_agent_create(listen_flag)
            print(GREEN + f"<<< USER >>> " + END)
        elif user_input_prompt.lower() == "/listen on":
            listen_flag = True
        elif user_input_prompt.lower() == "/listen off":
            listen_flag = False
            print(GREEN + f"<<< USER >>> " + END)
        elif user_input_prompt.lower() == "/leap on":
            leap_flag = True
            print(GREEN + f"<<< USER >>> " + END)
        elif user_input_prompt.lower() == "/leap off":
            leap_flag = False
            print(GREEN + f"<<< USER >>> " + END)
        elif user_input_prompt.lower() == "/speech on":
            leap_flag = False
            listen_flag = False
            print(GREEN + f"<<< USER >>> " + END)
        elif user_input_prompt.lower() == "/speech off":
            leap_flag = True
            listen_flag = True
            print(GREEN + f"<<< USER >>> " + END)

        elif speech_done == True:
            print(YELLOW + f"{user_input_prompt}" + OKCYAN)
            # Send the prompt to the assistant
            response = ollama_chatbot_class.send_prompt(user_input_prompt, user_input_model_select)
            print(RED + f"<<< {user_input_model_select} >>> " + END + f"{response}" + RED)

            # Preprocess for text to speech, add flag for if text to speech enable handle canche otherwise do /leap or smt
            # Clear speech cache and split the response into sentences for next TTS cache
            if leap_flag is not None and isinstance(leap_flag, bool):
                if leap_flag != True:
                    directory_manager_class.clear_directory(tts_processor_class.tts_store_wav_locker_path)
                    tts_processor_class.process_tts_responses(response)
            elif leap_flag is None:
                pass
            print(GREEN + f"<<< USER >>> " + END)

            # TODO Add commands for 0.21: 
            # /save as, /load as, - get done NOWWWWWW
            
            # TODO Add commands for 0.22: 
            # /speech - done, /listen - done, /leap - done

            # TODO Add commands for 0.23: 
            # /voice, /record, /clone voice, /playback, /music play, /movie play

            # TODO 0.25 RAG AND GOOGLE API

            # TODO LORA AND SORA STABLE DIFFUSION