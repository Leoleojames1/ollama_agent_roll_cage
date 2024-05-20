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
import json
import re
import keyboard
import time
import speech_recognition as sr
import ollama
import shutil
import threading

from tts_processor_class import tts_processor_class
from directory_manager_class import directory_manager_class
from latex_render_class import latex_render_class
from unsloth_train_class import unsloth_train_class

# from tensorflow.keras.models import load_model
# sentiment_model = load_model('D:\\CodingGit_StorageHDD\\model_git\\emotions_classifier\\emotions_classifier.keras')

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
        # TODO does this affect stt?
        self.url = "http://localhost:11434/api/chat"
        # self.url = "http://localhost:11434/api/generate"
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
        # self.chat_history.append({"role": "system", "content": "You are Rick, you are tasked with guiding morty from earth c137 on his adventures through the multiverse, here is morty. "})
        # self.chat_history.append({"role": "system", "content": "You are tasked with responding to the user, if the user requests for latex code utilize \[...\] formating, DO NOT USE $$...$$ latex formatting, otherwise respond to the user."})
                self.chat_history.append({"role": "system", "content": "You are tasked with responding to the user, if the user requests for latex code utilize \[...\] formating, DO NOT USE $$...$$ latex formatting, otherwise respond to the user."})
        self.chat_history.append({"role": "user", "content": user_input_prompt})
        data = {
            "model": self.user_input_model_select,
            "stream": False,
        }

        try:
            # print(f"{self.chat_history}")
            response = ollama.chat(model=self.user_input_model_select, messages=(self.chat_history), stream=False )
            # print(response)

        except Exception as e:
            return f"Error: {e}"

        # try:
        #     response_data = json.loads(response.text)
        # except json.JSONDecodeError:
        #     return "Error: Unable to parse response from model"

        if "message" in response:
            llama_response = response.get("message")
            self.chat_history.append(llama_response)
            return llama_response["content"]
        else:
            return "Error: Response from model is not in the expected format"

    # def send_prompt(self, user_input_prompt):
    #     """ a method for prompting the model
    #         args: user_input_prompt, user_input_model_select, search_google
    #         returns: none
    #     """
    #     self.chat_history.append({"user_name": "User", "message": user_input_prompt})

    #     # join chat history, 3 turns
    #     history = " ".join([message["message"] for message in self.chat_history[-3:]])

    #     # Use only the most recent user input as the prompt
    #     prompt = self.chat_history[-1]["message"]

    #     # Combine history and prompt
    #     full_prompt = history + "\n\nThis is the current prompt in the conversation. Do not respond to past conversation history only refer to it in the past tense, and certainly do not respond to yourself only respond to the User's current prompt here: " + prompt

    #     data = {
    #         "model": self.user_input_model_select,
    #         "stream": False,
    #         "prompt": full_prompt,
    #     }

    #     try:
    #         # Post the request to the model
    #         response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
    #         response.raise_for_status()  # Raise an exception if the status code is not 200
    #     except requests.RequestException as e:
    #         return f"Error: {e}"

    #     try:
    #         response_data = json.loads(response.text)
    #     except json.JSONDecodeError:
    #         return "Error: Unable to parse response from model"

    #     if "response" in response_data:
    #         llama_response = response_data.get("response")
            
    #         # Analyze the sentiment of the generated text
    #         sentiment = sentiment_model.predict([llama_response])
            
    #         # Combine the Llama response and sentiment analysis with a space
    #         combined_response = f"{llama_response} {sentiment}"
            
    #         self.chat_history.append({"model_name": f"{self.user_input_model_select}", "message": combined_response})
    #         return combined_response
    #     else:
    #         return "Error: Response from model is not in the expected format"
            
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
        # Check if user_input_model_select contains a slash
        if "/" in self.user_input_model_select:
            user_dir, model_dir = self.user_input_model_select.split("/")
            file_load_path_dir = os.path.join(self.conversation_library, user_dir, model_dir)
        else:
            file_load_path_dir = os.path.join(self.conversation_library, self.user_input_model_select)

        file_load_path_str = os.path.join(file_load_path_dir, f"{self.load_name}.json")
        directory_manager_class.create_directory_if_not_exists(file_load_path_dir)
        print(f"file path 1:{file_load_path_dir} \n")
        print(f"file path 2:{file_load_path_str} \n")
        with open(file_load_path_str, "r") as json_file:
            self.chat_history = json.load(json_file)

    def create_agent_cmd(self, user_create_agent_name, cmd_file_name):
        """Executes the create_agent_automation.cmd file with the specified agent name.
            Args: 
            Returns: None
        """
        try:
            # Construct the path to the create_agent_automation.cmd file
            batch_file_path = os.path.join(self.current_dir, cmd_file_name)

            # Call the batch file
            subprocess.run(f"call {batch_file_path} {user_create_agent_name}", shell=True)
        except Exception as e:
            print(f"Error executing create_agent_cmd: {str(e)}")
    
    def write_model_file_and_run_agent_create_ollama(self, listen_flag):
        """ a method to automatically generate a new agent via commands
            args: listen_flag
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
                self.user_create_agent_name = tts_processor_class.recognize_speech(mic_audio)
                self.user_create_agent_name = tts_processor_class.file_name_conversation_history_filter(self.user_create_agent_name)
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
            self.user_create_agent_name = input(WARNING + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + OKBLUE)
            user_input_temperature = input(WARNING + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + OKBLUE)
            system_prompt = input(WHITE + "<<< PROVIDE SYSTEM PROMPT >>> " + OKBLUE)

        model_create_dir = os.path.join(self.ignored_agents, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents, f"{self.user_create_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)
            # Get current model template data
            self.current_model_template()
            # Create the text file
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {self.user_input_model_select}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"\n#Set the system prompt\n")
                f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
                f.write(f"TEMPLATE \"\"\"\n{self.template}\n\"\"\"\n")

            # Execute create_agent_cmd
            self.create_agent_cmd(self.user_create_agent_name, 'create_agent_automation_ollama.cmd')
            return
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"

    def write_model_file_and_run_agent_create_gguf(self, listen_flag, model_git):
            """ a method to automatically generate a new agent via commands
                args: none
                returns: none
            """
            self.model_git = model_git
            # collect agent data with stt or ttt
            if listen_flag == False:  # listen_flag is False, use speech recognition
                print("Press space bar to record the new agent's agent name.")
                while not keyboard.is_pressed('space'):  # wait for space bar press
                    time.sleep(0.1)
                # Now start recording
                try:
                    mic_audio = tts_processor_class.get_audio()
                    self.user_create_agent_name = tts_processor_class.recognize_speech(mic_audio)
                    self.user_create_agent_name = tts_processor_class.file_name_conversation_history_filter(self.user_create_agent_name)
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
                self.converted_gguf_model_name = input(WARNING + "<<< PROVIDE SAFETENSOR CONVERTED GGUF NAME (without .gguf) >>> " + OKBLUE)
                self.user_create_agent_name = input(WARNING + "<<< PROVIDE NEW AGENT NAME TO CREATE >>> " + OKBLUE)
                user_input_temperature = input(WARNING + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + OKBLUE)
                system_prompt = input(WHITE + "<<< PROVIDE SYSTEM PROMPT >>> " + OKBLUE)

            model_create_dir = os.path.join(self.ignored_agents, f"{self.user_create_agent_name}")
            model_create_file = os.path.join(self.ignored_agents, f"{self.user_create_agent_name}\\modelfile")
            self.gguf_path_part = os.path.join(self.model_git, "converted")
            self.gguf_path = os.path.join(self.gguf_path_part, f"{self.converted_gguf_model_name}.gguf")
            print(f"model_git: {model_git}")
            try:
                # Create the new directory
                os.makedirs(model_create_dir, exist_ok=True)

                # Copy gguf to IgnoredAgents dir
                self.copy_gguf_to_ignored_agents()

                # Create the text file
                with open(model_create_file, 'w') as f:
                    f.write(f"FROM ./{self.converted_gguf_model_name}.gguf\n")
                    f.write(f"#temperature higher -> creative, lower -> coherent\n")
                    f.write(f"PARAMETER temperature {user_input_temperature}\n")
                    f.write(f"\n#Set the system prompt\n")
                    f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
                
                # Execute create_agent_cmd
                self.create_agent_cmd(self.user_create_agent_name, "create_agent_automation_gguf.cmd")
                return
            except Exception as e:
                return f"Error creating directory or text file: {str(e)}"

    def copy_gguf_to_ignored_agents(self):
        """ a method to setup the gguf before gguf to ollama create """
        self.create_ollama_model_dir = os.path.join(self.ignored_agents, self.user_create_agent_name)
        print(self.create_ollama_model_dir)
        print(self.gguf_path)
        print(self.create_ollama_model_dir)
        # Copy the file from self.gguf_path to create_ollama_model_dir
        shutil.copy(self.gguf_path, self.create_ollama_model_dir)
        return
    
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

        user_input_prompt = re.sub(r"activate latex on", "/latex on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate latex off", "/latex off", user_input_prompt, flags=re.IGNORECASE)

        user_input_prompt = re.sub(r"activate show model", "/show model", user_input_prompt, flags=re.IGNORECASE)

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

        # # Search for the name after 'forward slash voice swap'
        # match = re.search(r"(activate convert tensor|/convert tensor) ([^\s]*)", user_input_prompt, flags=re.IGNORECASE)
        # if match:
        #     self.tensor_name = match.group(2)

        return user_input_prompt 
    
    def current_model_template(self):
        """ a method for getting the model template
        """
        modelfile_data = ollama.show(f'{self.user_input_model_select}')
        for key, value in modelfile_data.items():
            if key == 'template':
                self.template = value
        return
        
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
    leap_flag = False
    listen_flag = True
    latex_flag = False

    # instantiate class calls
    tts_processor_class = tts_processor_class()
    directory_manager_class = directory_manager_class()
    unsloth_train_instance = unsloth_train_class()

    # begin chatbot loop
    ollama_chatbot_class.user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME >>> " + OKBLUE)
    # new instance class
    ollama_chatbot_class = ollama_chatbot_class(ollama_chatbot_class.user_input_model_select)
    latex_render_instance = None

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
                    print(">>AUDIO SENDING<<")
                    audio = tts_processor_class.get_audio()
                    print(">>AUDIO RECEIVED<<")
                    # Recognize speech to text from audio
                    user_input_prompt = tts_processor_class.recognize_speech(audio)
                    print(f">>SPEECH RECOGNIZED<< >> {user_input_prompt} <<")
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

        elif re.match(r"(activate voice swap|/voice swap) ([^/.]*)", user_input_prompt.lower()):
            print(f"Agent voice swapped to {ollama_chatbot_class.voice_name}")
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif re.match(r"(activate save as|/save as) ([^/.]*)", user_input_prompt.lower()):
            ollama_chatbot_class.save_to_json()
            print(f"Chat history saved to {ollama_chatbot_class.save_name}.json")
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif re.match(r"(activate load as|/load as) ([^/.]*)", user_input_prompt.lower()):
            ollama_chatbot_class.load_from_json()
            print(f"Chat history loaded from {ollama_chatbot_class.load_name}.json")
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif re.match(r"(activate convert tensor gguf|/convert tensor gguf) ([^\s]*)", user_input_prompt.lower()):  
            unsloth_train_instance.safe_tensor_gguf_convert(ollama_chatbot_class.tensor_name)
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif user_input_prompt.lower() == "/convert gguf ollama": 
            ollama_chatbot_class.write_model_file_and_run_agent_create_gguf(listen_flag, unsloth_train_instance.model_git)
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

        elif user_input_prompt.lower() == "/latex on":
            latex_flag = True
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif user_input_prompt.lower() == "/latex off":
            latex_flag = False
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif user_input_prompt.lower() == "/quit":
            break

        elif user_input_prompt.lower() == "/create":
            ollama_chatbot_class.write_model_file_and_run_agent_create_ollama(listen_flag)
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif user_input_prompt.lower() == "/function on":
            pass
            ollama_chatbot_class.function_call_model_select
            function_call_chatbot_class = ollama_chatbot_class(ollama_chatbot_class.function_call_model_select)
            response = function_call_chatbot_class.send_prompt(user_input_prompt)
            print(GREEN + f"<<< USER >>> " + OKGREEN)

        elif user_input_prompt.lower() == "/ollama show":
            modelfile_data = ollama.show(f'{ollama_chatbot_class.user_input_model_select}')
            for key, value in modelfile_data.items():
                if key != 'license':
                    print(RED + f"<<< {ollama_chatbot_class.user_input_model_select} >>> " + OKBLUE + f"{key}: {value}")

        elif user_input_prompt.lower() == "/ollama template":
            modelfile_data = ollama.show(f'{ollama_chatbot_class.user_input_model_select}')
            for key, value in modelfile_data.items():
                if key == 'template':
                    print(RED + f"<<< {ollama_chatbot_class.user_input_model_select} >>> " + OKBLUE + f"{key}: {value}")

        elif user_input_prompt.lower() == "/ollama license":
            modelfile_data = ollama.show(f'{ollama_chatbot_class.user_input_model_select}')
            for key, value in modelfile_data.items():
                if key == 'license':
                    print(RED + f"<<< {ollama_chatbot_class.user_input_model_select} >>> " + OKBLUE + f"{key}: {value}")

        elif user_input_prompt.lower() == "/ollama list":
            ollama_list = ollama.list()
            for model_info in ollama_list.get('models', []):
                model_name = model_info.get('name')
                model = model_info.get('model')
                print(RED + f"<<< {ollama_chatbot_class.user_input_model_select} >>> " + OKBLUE + f"{model_name}" + RED + " <<< ")

        elif speech_done == True:
            print(YELLOW + f"{user_input_prompt}" + OKCYAN)
            # Send the prompt to the assistant
            response = ollama_chatbot_class.send_prompt(user_input_prompt)
            print(RED + f"<<< {ollama_chatbot_class.user_input_model_select} >>> " + END + f"{response}" + RED)

            # Check for latex and add to queue
            if latex_flag:
                # Create a new instance
                latex_render_instance = latex_render_class()
                latex_render_instance.add_latex_code(response, ollama_chatbot_class.user_input_model_select)

            # Preprocess for text to speech, add flag for if text to speech enable handle canche otherwise do /leap or smt
            # Clear speech cache and split the response into sentences for next TTS cache
            if leap_flag is not None and isinstance(leap_flag, bool):
                if leap_flag != True:
                    # directory_manager_class.clear_directory(tts_processor_class.agent_voice_gen_dump)
                    tts_processor_class.process_tts_responses(response, ollama_chatbot_class.voice_name)
            elif leap_flag is None:
                pass
            # Start the mainloop in the main thread
            print(GREEN + f"<<< USER >>> " + END)