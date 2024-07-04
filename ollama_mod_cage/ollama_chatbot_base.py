""" ollama_chatbot_base.py

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
    All users have the right to develop and distribute ollama agent roll cage,
    with proper citation of the developers and repositories. Be weary that some
    software may not be licensed for commerical use.

    By: Leo Borcherding
        on github @ 
            leoleojames1/ollama_agent_roll_cage

"""
import os
import re
import time
import ollama
import base64
import keyboard
import pyaudio
import speech_recognition as sr
import curses
import threading
import json

from Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands
from Public_Chatbot_Base_Wand.speech_to_speech import tts_processor_class
from Public_Chatbot_Base_Wand.directory_manager import directory_manager_class
from Public_Chatbot_Base_Wand.latex_render import latex_render_class
from Public_Chatbot_Base_Wand.data_set_manipulator import data_set_constructor
from Public_Chatbot_Base_Wand.write_modelfile import model_write_class
from Public_Chatbot_Base_Wand.read_write_symbol_collector import read_write_symbol_collector
from Public_Chatbot_Base_Wand.data_set_manipulator import screen_shot_collector
from Public_Chatbot_Base_Wand.create_convert_model import create_convert_manager
from Public_Chatbot_Base_Wand.node_custom_methods import FileSharingNode
import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset
import sounddevice as sd

# TODO setup sebdg emotional classifyer keras 
# from tensorflow.keras.models import load_model
# sentiment_model = load_model('D:\\CodingGit_StorageHDD\\model_git\\emotions_classifier\\emotions_classifier.keras')

# -------------------------------------------------------------------------------------------------
class ollama_chatbot_base:
    """ 
    This class provides an interface to the Ollama local serve API for creating custom chatbot agents.
    It also provides access to Speech-to-Text transcription and Text-to-Speech generation methods via a low-level command line interface and the Tortoise TTS model.
    """

    # -------------------------------------------------------------------------------------------------
    # def __init__(self, agent_id, pad, win, lock, stdscr):
    def __init__(self):
        """ 
        Initialize the ollama_chatbot_base class with the given agent ID, model name, pad, window, and lock.

        Args:
            agent_id (int): The ID of the agent.
            model_name (str): The name of the model to use for the chatbot.
            pad (curses.pad): The pad to use for the chatbot's interface.
            win (curses.window): The window to use for the chatbot's interface.
            lock (threading.Lock): A threading lock to ensure thread-safe operations.
        """
        # initialize pads from curses
        # self.lock = lock
        self.pads = []
        # self.win = win
        # self.agent_id = agent_id
        self.user_input_model_select = None
        # self.pad = pad
        self.user_input_prompt = ""
        # self.stdscr = stdscr
        # get speech interrupt
        self.speech_interrupted = False
        
        # get user input model selection
        self.get_model()
        self.user_input_model_select = self.user_input_model_select

        # initialize chat
        self.chat_history = []
        self.llava_history = []

        # Default Agent Voice Reference
        #TODO add voice reference file manager
        # self.voice_name = "C3PO"
        self.voice_type = None
        self.voice_name = None

        # Default conversation name
        self.save_name = "default"
        self.load_name = "default"

        # TODO Connect api
        self.url = "http://localhost:11434/api/chat" #TODO REMOVE

        # Setup chat_history
        self.headers = {'Content-Type': 'application/json'}

        # get base path
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))

        # setup developer_tool.json
        self.developer_tools = os.path.abspath(os.path.join(self.current_dir, "developer_tools.json"))

        # get read write instance
        self.read_write_symbol_collector_instance = read_write_symbol_collector()

        # TODO if the developer tools file exists
        if hasattr(self, 'developer_tools'):
            self.developer_tools_dict = self.read_write_symbol_collector_instance.read_developer_tools_json()

        # setup base paths from developer tools path library
        self.ignored_agents = self.developer_tools_dict['ignored_agents_dir']
        self.llava_library = self.developer_tools_dict['llava_library_dir']
        self.model_git_dir = self.developer_tools_dict['model_git_dir']
        self.conversation_library = self.developer_tools_dict['conversation_library_dir']
        self.tts_voice_ref_wav_pack_path = self.developer_tools_dict['tts_voice_ref_wav_pack_path_dir']

        # build conversation save path #TODO ADD TO DEV DICT
        self.default_conversation_path = os.path.join(self.parent_dir, f"AgentFiles\\Ignored_pipeline\\conversation_library\\{self.user_input_model_select}\\{self.save_name}.json")

        # TEXT SECTION:
        self.latex_flag = False
        self.cmd_run_flag = None

        # SPEECH SECTION:
        self.leap_flag = True # TODO TURN OFF FOR MINECRAFT
        self.listen_flag = False # TODO TURN ON FOR MINECRAFT
        self.chunk_flag = False
        self.auto_speech_flag = False #TODO KEEP OFF BY DEFAULT FOR MINECRAFT, TURN ON TO START

        # VISION SECTION:
        self.llava_flag = False # TODO TURN ON FOR MINECRAFT
        self.splice_flag = False
        self.screen_shot_flag = False

        # ollama chatbot base setup wand class instantiation
        self.ollama_command_instance = ollama_commands(self.user_input_model_select, self.developer_tools_dict)
        self.colors = self.ollama_command_instance.colors
        self.directory_manager_class = directory_manager_class()
        # get data
        self.screen_shot_collector_instance = screen_shot_collector(self.developer_tools_dict)
        # splice data
        self.data_set_video_process_instance = data_set_constructor(self.developer_tools_dict)
        # generate
        self.model_write_class_instance = model_write_class(self.colors, self.developer_tools_dict)
        self.create_convert_manager_instance = create_convert_manager(self.colors, self.developer_tools_dict)
        # peer2peer node
        self.FileSharingNode_instance = FileSharingNode(host="127.0.0.1", port=9876)
        
    # -------------------------------------------------------------------------------------------------  
    def get_model(self):
        """ a method for collecting the model name from the user input
        """
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        self.user_input_model_select = input(HEADER + "<<< PROVIDE MODEL NAME >>> " + OKBLUE)
    
    # -------------------------------------------------------------------------------------------------
    def swap(self):
        """ a method to call when swapping models
        """
        self.chat_history = []
        self.user_input_model_select = input(self.colors['HEADER']+ "<<< PROVIDE AGENT NAME TO SWAP >>> " + self.colors['OKBLUE'])
        print(f"Model changed to {self.user_input_model_select}")
        return
    
    # -------------------------------------------------------------------------------------------------   
    def get_audio(self):
        """ a method for getting the user audio from the microphone
            args: none
        """
        print(">>AUDIO RECORDING<<")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []

        while self.auto_speech_flag and not self.chunk_flag:
            data = stream.read(1024)
            frames.append(data)

        print(">>AUDIO RECEIVED<<")
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert the audio data to an AudioData object
        audio = sr.AudioData(b''.join(frames), 16000, 2)
        self.chunk_flag = False  # Set chunk_flag to False here to indicate that the audio has been received
        return audio
    
    # -------------------------------------------------------------------------------------------------   
    def recognize_speech(self, audio):
        """ a method for calling the speech recognizer
            args: audio
            returns: speech_str
        """
        #TODO Realized current implementation calls google API, must replace with LOCAL SPEECH RECOGNITION MODEL
        speech_str = sr.Recognizer().recognize_google(audio)
        print(f">>{speech_str}<<")
        return speech_str
    
    # -------------------------------------------------------------------------------------------------      
    def system_prompt_manager(self, sys_prompt_select):
        """ a method for managing the current system prompt when called, based on the user_input_model_select 
            for automated fast shot prompting.
            Args: sys_prompt_select
            Returns: sys+prompt_select
        """
        # text model system prompts
        self.sys_prompts = {
            "borch/phi3_speed_chat" : "You are borch/phi3_speed_chat, a phi3 large language model, specifically you have been tuned to respond in a more quick and conversational manner, the user is using speech to text for communication, its also okay to be fun and wild as a phi3 ai assistant. Its also okay to respond with a question, if directed to do something just do it, and realize that not everything needs to be said in one shot, have a back and forth listening to the users response. If the user decides to request a latex math code output, use \[...\] instead of $$...$$ notation, if the user does not request latex, refrain from using latex unless necessary. Do not re-explain your response in a parend or bracketed note: the response... this is annoying and users dont like it.",
            "Minecraft" : "You are a helpful minecraft assistant, given the provided screenshot data please direct the user immediatedly, prioritize the order in which to inform the player, hostile mobs should be avoided or terminated, danger is a top priority, but so is crafting and building, if they require help quickly guide them to a solution in real time. Please respond in a quick conversational voice, do not read off of documentation, you need to directly explain quickly and effectively whats happening, for example if there is a zombie say something like, watch out thats a Zombie hurry up and kill it or run away, they are dangerous. The recognized Objects around the perimeter are usually items, health, hunger, breath, gui elements, or status affects, please differentiate these objects in the list from 3D objects in the forward facing perspective with hills trees, mobs etc, the items are held by the player and due to the perspective take up the warped edge of the image on the sides. the sky is typically up with a sun or moon and stars, with the dirt below, there is also the nether which is a firey wasteland and cave systems with ore. Please stick to whats relevant to the current user prompt and llava data:"
            #TODO add text prompts for the following ideas:
            # latex pdf book library rag
            # c3po adventure
            # rick and morty adveture
            # phi3 & llama3 fast shot prompting 
            # linked in, redbubble, oarc - advertising server api for laptop
        }

        # llava system prompts
        self.llava_sys_prompts = {
            "phi3" : "You are a helpful phi3-vision assistant, please describe the screen share being sent to you from the OARC user, they are requesting image recognition from you:",
            "Minecraft_llava_sys" : "You are a minecraft llava image recognizer, search for passive mobs, hostile mobs, trees, hills, blocks, and items, given the provided screenshot please provide a dictionary of the objects recognized paired with key attributed about each object, and only 1 sentence to describe anything else that is not captured by the dictionary, do not use more sentences, only list objects with which you have high confidence of recognizing and for low confidence describe shape and object type more heavily to gage hard recognitions. Objects around the perimeter are usually player held items like swords or food, gui elements like items, health, hunger, breath, or status affects, please differentiate these objects in the list from the 3D landscape objects in the forward facing perspective, the items are held by the player traversing the world and can place and remove blocks. Return dictionary and 1 summary sentence:",
            "Minecraft_llava_prompt" : "given the provided screenshot please provide a dictionary of key value pairs for each object in with image with its relative position, do not use sentences, if you cannot recognize the enemy describe the color and shape as an enemy in the dictionary"
            #TODO add text prompts for the following ideas:
        }

        # llava fast shot prompts
        self.llava_intermediate_prompts = {
            "phi3_Minecraft_prompt": "Based on the information in LLAVA_DATA please direct the user immediatedly, prioritize the order in which to inform the player of the identified objects, items, hills, trees and passive and hostile mobs etc. Do not output the dictionary list, instead conversationally express what the player needs to do quickly so that they can ask you more questions.",
            #TODO add text prompts for the following ideas:
        }

        # if user model select is in system prompt dictionaries, append fast shot to the chat accordingly
        if sys_prompt_select in self.sys_prompts:
            self.chat_history.append({"role": "system", "content": self.sys_prompts[sys_prompt_select]})
        elif sys_prompt_select in self.llava_sys_prompts:
            self.chat_history.append({"role": "system", "content": self.llava_sys_prompts[sys_prompt_select]})
        elif sys_prompt_select in self.llava_intermediate_prompts:
            self.chat_history.append({"role": "system", "content": self.llava_intermediate_prompts[sys_prompt_select]})
        else:
            print("Invalid choice. Please select a valid prompt.")
        return

    # -------------------------------------------------------------------------------------------------   
    def llava_prompt_manager(self, sys_prompt_select):
        if sys_prompt_select in self.prompts:
            self.chat_history.append({"role": "system", "content": self.prompts[sys_prompt_select]})
        else:
            print("Invalid choice. Please select a valid prompt.")
        return sys_prompt_select
    
    # -------------------------------------------------------------------------------------------------
    def shot_prompt(self, prompt):
        # Clear chat history
        self.shot_history = []

        # Append user prompt
        self.chat_history.append({"role": "user", "content": prompt})

        try:
            response = ollama.generate(model=self.user_input_model_select, messages=self.shot_history, stream=False)
            if isinstance(response, dict) and "message" in response:
                model_response = response.get("message")
                self.chat_history.append(model_response)
                return model_response["content"]
            else:
                return "Error: Response from model is not in the expected format"
        except Exception as e:
            return f"Error: {e}"
    
    # -------------------------------------------------------------------------------------------------   
    def send_prompt(self, user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        #TODO ADD IF MEM OFF CLEAR HISTORY
        # self.chat_history = []
        #TODO ADD screen shot {clock & manager}
        self.screenshot_path = os.path.join(self.llava_library, "screenshot.png")

        # start prompt shot if flag is True TODO setup modular custom prompt selection
        self.prompt_shot_flag = False # TODO SETUP FLAG LOGIC
        if self.prompt_shot_flag is True:
            sys_prompt_select = f"{self.user_input_model_select}"
            self.system_prompt_manager(sys_prompt_select)

        # append user prompt
        self.chat_history.append({"role": "user", "content": user_input_prompt})

        # get the llava response and append it to the chat history only if an image is provided
        if self.llava_flag is True:
            # load the screenshot and convert it to a base64 string
            with open(f'{self.screenshot_path}', 'rb') as f:
                user_screenshot_raw2 = base64.b64encode(f.read()).decode('utf-8')
                self.user_screenshot_raw = user_screenshot_raw2
            #TODO manage user_input_prompt for llava model during conversation
            llava_response = self.llava_prompt(user_screenshot_raw2, user_input_prompt)
            print(f"LLAVA SOURCE: {llava_response}")
            # TODO DOES THIS DO ANYTHING? I DONT THINK SO
            self.chat_history.append({"role": "assistant", "content": f"LLAVA_DATA: {llava_response}"})
            self.chat_history.append({"role": "user", "content": "Based on the information in LLAVA_DATA please direct the user immediatedly, prioritize the order in which to inform the player of the identified objects, items, hills, trees and passive and hostile mobs etc. Do not output the dictionary list, instead conversationally express what the player needs to do quickly so that they can ask you more questions."})

        try:
            response = ollama.chat(model=self.user_input_model_select, messages=(self.chat_history), stream=False )
            if isinstance(response, dict) and "message" in response:
                model_response = response.get("message")
                self.chat_history.append(model_response)
                return model_response["content"]
            else:
                return "Error: Response from model is not in the expected format"
        except Exception as e:
            return f"Error: {e}"
        
    # -------------------------------------------------------------------------------------------------   
    def llava_prompt(self, user_screenshot_raw2, llava_user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        self.llava_user_input_prompt = llava_user_input_prompt
        self.llava_history = []
        self.llava_history.append({"role": "system", "content": "You are a minecraft llava image recognizer, search for passive mobs, hostile mobs, trees, hills, blocks, and items, given the provided screenshot please provide a dictionary of the objects recognized paired with key attributed about each object, and only 1 sentence to describe anything else that is not captured by the dictionary, do not use more sentences, only list objects with which you have high confidence of recognizing and for low confidence describe shape and object type more heavily to gage hard recognitions. Objects around the perimeter are usually player held items like swords or food, gui elements like items, health, hunger, breath, or status affects, please differentiate these objects in the list from the 3D landscape objects in the forward facing perspective, the items are held by the player traversing the world and can place and remove blocks. Return dictionary and 1 summary sentence:"})
        message = {"role": "user", "content": "given the provided screenshot please provide a dictionary of key value pairs for each object in with image with its relative position, do not use sentences, if you cannot recognize the enemy describe the color and shape as an enemy in the dictionary"}

        image_message = None
        if user_screenshot_raw2 is not None:
            # Assuming user_input_image is a base64 encoded image
            message["images"] = [user_screenshot_raw2]
            image_message = message
        try:
            response_llava = ollama.chat(model="llava", messages=(self.llava_history + [image_message]), stream=False )
        except Exception as e:
            return f"Error: {e}"

        if "message" in response_llava:
            
            # print(f"LAVA_RECOGNITION: {message}")
            model_response = response_llava.get("message")
            self.llava_history.append({"role": "assistant", "content": model_response["content"]})
            # print(f"LLAVA HISTORY: {self.llava_history}")

            # Keep only the last 2 responses in llava_history
            self.llava_history = self.llava_history[-2:]

            return model_response["content"]
        else:
            return "Error: Response from model is not in the expected format"
        
    # -------------------------------------------------------------------------------------------------
    def voice_command_select_filter(self, user_input_prompt):
        """ a method for managing the voice command selection
            Args: user_input_prompt
            Returns: user_input_prompt
        """ 
        # Parse for general commands (non token specific args)
        user_input_prompt = re.sub(r"activate swap", "/swap", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate quit", "/quit", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate llama create", "/llama create", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate listen on", "/listen on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate listen on", "/listen off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate speech on", "/speech on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate speech off", "/speech off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate leap on", "/leap on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate leap off", "/leap off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate latex on", "/latex on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate latex off", "/latex off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate show model", "/llama show", user_input_prompt, flags=re.IGNORECASE)

        # Parse for Token Specific Arg Commands
        # # Parse for the name after 'forward slash voice swap'
        # match = re.search(r"(activate voice swap|/voice swap) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        # if match:
        #     self.voice_name = match.group(2)
        #     self.voice_name = self.tts_processor_instance.file_name_conversation_history_filter(self.voice_name)

        # Parse for the name after 'forward slash movie'
        match = re.search(r"(activate movie|/movie) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.movie_name = match.group(2)
            self.movie_name = self.file_name_conversation_history_filter(self.movie_name)
        else:
            self.movie_name = None

        # Parse for the name after 'activate save'
        match = re.search(r"(activate save as|/save as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.save_name = match.group(2)
            self.save_name = self.file_name_conversation_history_filter(self.save_name)
            print(f"save_name string: {self.save_name}")
        else:
            self.save_name = None

        # Parse for the name after 'activate load'
        match = re.search(r"(activate load as|/load as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.load_name = match.group(2)
            self.load_name = self.file_name_conversation_history_filter(self.load_name)
            print(f"load_name string: {self.load_name}")
        else:
            self.load_name = None

        # Parse for the name after 'forward slash voice swap'
        match = re.search(r"(activate convert tensor|/convert tensor) ([^\s]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.tensor_name = match.group(2)

        return user_input_prompt
    
    # -------------------------------------------------------------------------------------------------
    def command_select(self, command_str):
        """ 
            Parse user_input_prompt as command_str to see if their is a command to select & execute for the current chatbot instance

            Args: command_str
            Returns: command_library[command_str]
        """
        command_library = {
            "/swap": lambda: self.swap(),
            "/voice swap": lambda: self.voice_swap(),
            "/save as": lambda: self.save_to_json(self.save_name, self.user_input_model_select),
            "/load as": lambda: self.load_from_json(self.load_name, self.user_input_model_select),
            "/write modelfile": lambda: self.model_write_class_instance.write_model_file(),
            "/convert tensor": lambda: self.create_convert_manager_instance.safe_tensor_gguf_convert(self.tensor_name),
            "/convert gguf": lambda: self.model_write_class_instance.write_model_file_and_run_agent_create_gguf(self.listen_flag, self.model_git),
            "/listen on": lambda: self.listen(True),
            "/listen off": lambda: self.listen(False),
            "/leap on": lambda: self.leap(True),
            "/leap off": lambda: self.leap(False),
            "/speech on": lambda: self.speech(False, True),
            "/speech off": lambda: self.speech(True, False),
            "/latex on": lambda: self.latex(True),
            "/latex off": lambda: self.latex(False),
            "/command auto on": lambda: self.auto_commands(True),
            "/command auto off": lambda: self.auto_commands(False),
            "/llava flow": lambda: self.llava_flow(True),
            "/llava freeze": lambda: self.llava_flow(False),
            "/auto on": lambda: self.auto_speech_set(True),
            "/auto off": lambda: self.auto_speech_set(False),
            "/quit": lambda: self.ollama_command_instance.quit(),
            "/llama create": lambda: self.ollama_command_instance.ollama_create(),
            "/llama show": lambda: self.ollama_command_instance.ollama_show_modelfile(),
            "/llama template": lambda: self.ollama_command_instance.ollama_show_template(),
            "/llama license": lambda: self.ollama_command_instance.ollama_show_license(),
            "/llama list": lambda: self.ollama_command_instance.ollama_list(),
            "/splice video": lambda: self.data_set_video_process_instance.generate_image_data(),
            "/developer new" : lambda: self.read_write_symbol_collector_instance.developer_tools_generate(),
            "/start node": lambda: self.FileSharingNode_instance.start_node(),
            "/synthetic generator": lambda: self.generate_synthetic_data(),
            "/convert wav": lambda: self.data_set_video_process_instance.call_convert()
        }
        
        # Find the command in the command string
        command = next((cmd for cmd in command_library.keys() if command_str.startswith(cmd)), None)

        # If a command is found, split it from the arguments
        if command:
            args = command_str[len(command):].strip()
        else:
            args = None

        # Check if the command is in the library, if not return None
        if command in command_library:
            command_library[command]()
            cmd_run_flag = True
            return cmd_run_flag
        else:
            cmd_run_flag = False
            return cmd_run_flag
        
    # -------------------------------------------------------------------------------------------------
    def file_name_conversation_history_filter(self, input):
        """ a method for preprocessing the voice recognition with a filter before forwarding the agent file names.
            args: user_input_agent_name
            returns: user_input_agent_name
        """
        # Use regex to replace all spaces with underscores and convert to lowercase
        output = re.sub(' ', '_', input).lower()
        return output
    
    # -------------------------------------------------------------------------------------------------
    def file_name_voice_filter(self, input):
        """ a method for preprocessing the voice recognition with a filter before forwarding the agent file names.
            args: user_input_agent_name
            returns: user_input_agent_name
        """
        # Use regex to replace all spaces with underscores
        output = re.sub(' ', '_', input).lower()
        return output
    
    # -------------------------------------------------------------------------------------------------
    def get_available_voices(self):
        # Get list of fine-tuned models
        fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
        fine_tuned_voices = [d[8:] for d in os.listdir(fine_tuned_dir) if os.path.isdir(os.path.join(fine_tuned_dir, d)) and d.startswith("XTTS-v2_")]
        
        # Get list of voice reference samples
        reference_voices = [d for d in os.listdir(self.tts_voice_ref_wav_pack_path) if os.path.isdir(os.path.join(self.tts_voice_ref_wav_pack_path, d))]
        
        return fine_tuned_voices, reference_voices
    
    # -------------------------------------------------------------------------------------------------
    def get_voice_selection(self):
        print("Available voices:")
        fine_tuned_voices, reference_voices = self.get_available_voices()
        all_voices = fine_tuned_voices + reference_voices
        for i, voice in enumerate(all_voices):
            print(f"{i + 1}. {voice}")
        
        while True:
            selection = input("Select a voice (enter the number): ")
            try:
                index = int(selection) - 1
                if 0 <= index < len(all_voices):
                    selected_voice = all_voices[index]
                    if selected_voice in fine_tuned_voices:
                        self.voice_name = selected_voice
                        self.voice_type = "fine_tuned"
                    else:
                        self.voice_name = selected_voice
                        self.voice_type = "reference"
                    return
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    # -------------------------------------------------------------------------------------------------   
    def save_to_json(self, save_name, user_input_model_select):
        """ a method for saving the current agent conversation history
            Args: filename
            Returns: none
        """
        self.save_name = save_name
        self.user_input_model_select = user_input_model_select
        file_save_path_dir = os.path.join(self.conversation_library, f"{self.user_input_model_select}")
        file_save_path_str = os.path.join(file_save_path_dir, f"{self.save_name}.json")
        self.directory_manager_class.create_directory_if_not_exists(file_save_path_dir)
        
        print(f"file path 1:{file_save_path_dir} \n")
        print(f"file path 2:{file_save_path_str} \n")
        with open(file_save_path_str, "w") as json_file:
            json.dump(self.chat_history, json_file, indent=2)

    # -------------------------------------------------------------------------------------------------   
    def load_from_json(self, load_name, user_input_model_select):
        """ a method for loading the directed conversation history to the current agent, mis matching
        agents and history may be bizarre
            Args: filename
            Returns: none
        """
        self.load_name = load_name
        self.user_input_model_select = user_input_model_select

        # Check if user_input_model_select contains a slash
        if "/" in self.user_input_model_select:
            user_dir, model_dir = self.user_input_model_select.split("/")
            file_load_path_dir = os.path.join(self.conversation_library, user_dir, model_dir)
        else:
            file_load_path_dir = os.path.join(self.conversation_library, self.user_input_model_select)

        file_load_path_str = os.path.join(file_load_path_dir, f"{self.load_name}.json")
        self.directory_manager_class.create_directory_if_not_exists(file_load_path_dir)
        print(f"file path 1:{file_load_path_dir} \n")
        print(f"file path 2:{file_load_path_str} \n")
        with open(file_load_path_str, "r") as json_file:
            self.chat_history = json.load(json_file)

    # # -------------------------------------------------------------------------------------------------
    # def print_to_win(self, message):
    #     """
    #     Print the given message to the chatbot's window and refresh the window.

    #     Args:
    #         message (str): The message to print.
    #     """
    #     if self.win is None:
    #         raise ValueError("self.win is None")
    #     if not isinstance(message, str):
    #         raise TypeError("message must be a string")
    #     try:
    #         self.win.addstr(message)
    #         self.win.refresh()
    #     except curses.error as e:
    #         raise RuntimeError(f"An error occurred while printing to the window: {e}") from e

    # # -------------------------------------------------------------------------------------------------
    # def update_pad(self):
    #     """
    #     Continuously update the chatbot's pad with the user's input and refresh the pad.
    #     """
    #     while True:
    #         if self.user_input_prompt:
    #             try:
    #                 with self.lock:  # Acquire the lock before updating the UI
    #                     self.pad.addstr(self.user_input_prompt)  # Add the user input to the pad
    #                     self.win.refresh()
    #                     self.pad.refresh(0, 0, 0, 0, self.win.getmaxyx()[0], self.win.getmaxyx()[1])  # Refresh the pad after adding the user's input
    #                     self.user_input_prompt = ""
    #             except curses.error as e:
    #                 print(f"An error occurred while updating the pad: {e}")

    # # -------------------------------------------------------------------------------------------------
    # def capture_keys(self):
    #     """
    #     Continuously capture keys from the user and update the user's input accordingly.
    #     """
    #     user_input = ""
    #     self.win.nodelay(True)
    #     while True:
    #         key = self.win.getch()
    #         if key == ord('\n'):
    #             self.user_input_prompt = user_input
    #             print(f"user_input_prompt is set to: {user_input}")
    #             user_input = ""
    #             self.update_pad()  # Update the pad after a newline is entered
    #         elif key == curses.KEY_BACKSPACE:
    #             user_input = user_input[:-1]
    #         elif key >= ord(' ') and key <= ord('~'):
    #             user_input += chr(key)
    #         elif key == -1:
    #             continue

    # # -------------------------------------------------------------------------------------------------
    # def handle_resize(self):
    #     """
    #     Handles screen resizing. Updates the size and position of each chatbot's
    #     window and pad to fit the new screen size.

    #     Args:
    #         None
    #     Returns:
    #         None
    #     """
    #     if self.chatbot is None:
    #         return

    #     max_y, max_x = self.stdscr.getmaxyx()
    #     win_height = max_y
    #     win_width = max_x - 2
    #     win_y = 0
    #     win_x = 1
    #     try:
    #         self.chatbot.win.resize(win_height, win_width)
    #         self.chatbot.win.mvwin(win_y, win_x)
    #         self.chatbot.pad.resize(100, win_width)
    #     except curses.error as e:
    #         self.stdscr.addstr(0, 0, f"Resize error: {e}\n")
    #         self.stdscr.refresh()

    # # -------------------------------------------------------------------------------------------------
    # def highlight_current_chatbot(self):
    #     """
    #     Highlights the currently selected chatbot by drawing a box around its window.
    #     All other chatbots are unhighlighted.

    #     Args:
    #         None

    #     Returns:
    #         None
    #     """
    #     # Only highlight if a chatbot instance is currently selected
    #     if self.chatbot is not None:
    #         # Set the color of the chatbot window to highlight
    #         self.chatbot.win.attron(curses.color_pair(2))
    #         # Draw a box around the chatbot window
    #         self.chatbot.win.box()
    #         # Turn off the highlight color
    #         self.chatbot.win.attroff(curses.color_pair(2))
    #         # Refresh the screen after updating the frame
    #         self.stdscr.refresh()  # Refresh the screen after updating the frame
            
    # # -------------------------------------------------------------------------------------------------
    # def chatbot_main(self):
    #     """
    #     The main loop for the chatbot. This method manages the chatbot instance, handling user input and chatbot responses.
    #     """

    #     try:
    #         # Get the model name
    #         self.stdscr.addstr("<<< PROVIDE MODEL NAME >>> ", curses.color_pair(1))
    #         self.stdscr.refresh()
    #         self.user_input_model_select = self.stdscr.getstr().decode('utf-8')

    #         print(f"chatbot_main is being called for {self.user_input_model_select}")

    #         # Start the update method in a separate thread
    #         update_pad_thread = threading.Thread(target=self.update_pad)
    #         update_pad_thread.start()

    #         self.latex_render_instance = None
    #         self.tts_processor_instance = None

    #         self.print_to_win("Press space bar to record audio:")
    #         self.print_to_win("<<< USER >>> ")

    #         keyboard.add_hotkey('ctrl+w', self.auto_speech_set, args=(True,))
    #         keyboard.add_hotkey('ctrl+s', self.chunk_speech, args=(True,))

    #         # Start the capture_keys method in a separate thread
    #         capture_keys_thread = threading.Thread(target=self.capture_keys)
    #         capture_keys_thread.start()

    #         while True:
    #             user_input_prompt = ""
    #             speech_done = False
    #             cmd_run_flag = False

    #             if self.listen_flag or self.auto_speech_flag:
    #                 self.tts_processor_instance = self.instance_tts_processor()
    #                 while self.auto_speech_flag:  # user holds down the space bar
    #                     try:
    #                         # Record audio from microphone
    #                         audio = self.get_audio()
    #                         if self.listen_flag:
    #                             # Recognize speech to text from audio
    #                             user_input_prompt = self.recognize_speech(audio)
    #                             self.print_to_win(f">>SPEECH RECOGNIZED<< >> {user_input_prompt} <<")
    #                             speech_done = True
    #                             self.chunk_flag = False
    #                             self.print_to_win(f"CHUNK FLAG STATE: {self.chunk_flag}")
    #                             self.auto_speech_flag = False
    #                     except sr.UnknownValueError:
    #                         self.print_to_win(self.colors["OKCYAN"] + "Google Speech Recognition could not understand audio" + self.colors["OKCYAN"])
    #                     except sr.RequestError as e:
    #                         self.print_to_win(self.colors["OKCYAN"] + f"Could not request results from Google Speech Recognition service: {e}" + self.colors["OKCYAN"])
    #             elif not self.listen_flag:
    #                 self.print_to_win(self.colors["OKCYAN"] + "Please type your selected prompt:" + self.colors["OKCYAN"])
    #                 user_input_prompt = input(self.colors["GREEN"] + f"<<< USER >>> " + self.colors["END"])
    #                 speech_done = True
    #             user_input_prompt = self.voice_command_select_filter(user_input_prompt)
    #             cmd_run_flag = self.command_select(user_input_prompt)
    #             # get screenshot
    #             if self.llava_flag:
    #                 self.screen_shot_flag = self.screen_shot_collector_instance.get_screenshot()
    #             # splice videos
    #             if self.splice_flag:
    #                 self.data_set_video_process_instance.generate_image_data()
    #             if not cmd_run_flag and speech_done:
    #                 self.print_to_win(f"{user_input_prompt}\n")
    #                 # Send the prompt to the assistant
    #                 if self.screen_shot_flag:
    #                     response = self.send_prompt(user_input_prompt)
    #                     self.screen_shot_flag = False
    #                 else:
    #                     response = self.send_prompt(user_input_prompt)
    #                 self.print_to_win(f"<<< {self.user_input_model_select} >>> {response}\n")

    #                 if self.latex_flag:
    #                     # Create a new instance
    #                     latex_render_instance = latex_render_class()
    #                     latex_render_instance.add_latex_code(response, self.user_input_model_select)
    #                 # Preprocess for text to speech, add flag for if text to speech enable handle canche otherwise do /leap or smt
    #                 # Clear speech cache and split the response into sentences for next TTS cache
    #                 if self.leap_flag is not None and not self.leap_flag:
    #                     self.tts_processor_instance.process_tts_responses(response, self.voice_name)
    #                 elif self.leap_flag is None:
    #                     pass
    #                 # Start the mainloop in the main thread
    #                 self.print_to_win(self.colors["GREEN"] + f"<<< USER >>> " + self.colors["END"])
    #     except Exception as e:
    #         print(f"An error occurred in the thread for {self.user_input_model_select}: {e}")

    # -------------------------------------------------------------------------------------------------   
    def chunk_speech(self, value):
        """
        This method sets the chunk_flag to the given value and prints its state.
        The chunk_flag is used to control whether the speech input should be chunked.

        Args:
            value (bool): The value to set the chunk_flag to.
        """
        # time.sleep(1)
        self.chunk_flag = value
        print(f"chunk_flag FLAG STATE: {self.chunk_flag}")

    # -------------------------------------------------------------------------------------------------   
    def auto_speech_set(self, value):
        """
        This method sets the auto_speech_flag and chunk_flag to the given value and False respectively, and prints the state of auto_speech_flag.
        The auto_speech_flag is used to control whether the speech input should be automatically processed.

        Args:
            value (bool): The value to set the auto_speech_flag to.
        """
        self.auto_speech_flag = value
        self.chunk_flag = False
        print(f"auto_speech_flag FLAG STATE: {self.auto_speech_flag}")

    # -------------------------------------------------------------------------------------------------
    def instance_tts_processor(self, voice_type, voice_name):
        """
        This method creates a new instance of the tts_processor_class if it doesn't already exist, and returns it.
        The tts_processor_class is used for processing text-to-speech responses.

        Returns:
            tts_processor_instance (tts_processor_class): The instance of the tts_processor_class.
        """
        if not hasattr(self, 'tts_processor_instance') or self.tts_processor_instance is None:
            self.tts_processor_instance = tts_processor_class(self.colors, self.developer_tools_dict, voice_type, voice_name)
        return self.tts_processor_instance
    
    # -------------------------------------------------------------------------------------------------   
    def leap(self, flag):
        """ a method for changing the leap flag 
            args: flag
            returns: none
        """
        #TODO add ERROR handling
        if flag == True:
            print(self.colors["OKBLUE"] + "- text to speech deactivated -" + self.colors["RED"])
        self.leap_flag = flag
        if flag == False:
            print(self.colors["OKBLUE"] + "- text to speech activated -" + self.colors["RED"])
            self.get_voice_selection()
            self.tts_processor_instance = self.instance_tts_processor(self.voice_type, self.voice_name)
        print(f"leap_flag FLAG STATE: {self.leap_flag}")
        return
    
    # -------------------------------------------------------------------------------------------------   
    def speech(self, flag1, flag2):
        """ a method for changing the speech to speech flags 
            args: flag1, flag2
            returns: none
        """
        #TODO add ERROR handling
        if flag2 == False:
            print(self.colors["OKBLUE"] + "- speech to text deactivated -" + self.colors["RED"])
            print(self.colors["OKBLUE"] + "- text to speech deactivated -" + self.colors["RED"])
        if flag2 == True:
            print(self.colors["OKBLUE"] + "- speech to text activated -" + self.colors["RED"])
            print(self.colors["OKCYAN"] + "🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️" + self.colors["OKCYAN"])
            print(self.colors["OKBLUE"] + "- text to speech activated -" + self.colors["RED"])
            self.get_voice_selection()
            self.tts_processor_instance = self.instance_tts_processor(self.voice_type, self.voice_name)
        self.leap_flag = flag1
        self.listen_flag = flag2
        print(f"listen_flag FLAG STATE: {self.listen_flag}")
        print(f"leap_flag FLAG STATE: {self.leap_flag}")
        return
    # -------------------------------------------------------------------------------------------------   
    def latex(self, flag):
        """ a method for changing the latex render gui flag 
            args: flag
            returns: none
        """
        self.latex_flag = flag
        print(f"latex_flag FLAG STATE: {self.latex_flag}")        
        return
    
    # -------------------------------------------------------------------------------------------------   
    def llava_flow(self, flag):
        """ a method for changing the llava image recognition flag 
            args: flag
            returns: none
        """
        self.llava_flag = flag
        print(f"llava_flag FLAG STATE: {self.llava_flag}")
        return
    
    # -------------------------------------------------------------------------------------------------   
    def voice_swap(self):
        """ a method to call when swapping voices
            args: none
            returns: none
        """
        # Search for the name after 'forward slash voice swap'
        print(f"Agent voice swapped to {self.voice_name}")
        print(self.colors['GREEN'] + f"<<< USER >>> " + self.colors['OKGREEN'])
        return
    
    # -------------------------------------------------------------------------------------------------   
    def listen(self, flag):
        """ a method for changing the listen flag 
            args: flag
            return: none
        """
        self.listen_flag = flag
        print(f"listen_flag FLAG STATE: {self.listen_flag}")

        if flag == False:
            print(self.colors["OKBLUE"] + "- speech to text deactivated -" + self.colors["RED"])

        if flag == True:
            print(self.colors["OKBLUE"] + "- speech to text activated -" + self.colors["RED"])
            print(self.colors["OKCYAN"] + "🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️" + self.colors["OKCYAN"])
        return

    # -------------------------------------------------------------------------------------------------   
    def auto_commands(self, flag):
        """ a method for auto_command flag 
            args: flag
            return: none
        """
        self.auto_commands_flag = flag
        print(f"auto_commands FLAG STATE: {self.auto_commands_flag}")
        return
    
    # -------------------------------------------------------------------------------------------------
    def chatbot_main(self):
        """ a method for managing the current chatbot instance loop 
            args: None
            returns: None
        """

        # wait to load tts & latex until needed
        self.latex_render_instance = None
        self.tts_processor_instance = None
        # self.FileSharingNode = None

        # print(self.colors["OKCYAN"] + "🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️" + self.colors["OKCYAN"])

        # keyboard.add_hotkey('ctrl+shift', self.auto_speech_set, args=(True,))
        # keyboard.add_hotkey('ctrl+alt', self.chunk_speech, args=(True,))
        # keyboard.add_hotkey('shift+alt', self.interrupt_speech)
        while True:
            user_input_prompt = ""
            speech_done = False
            cmd_run_flag = False

            if self.listen_flag | self.auto_speech_flag is True:
                # self.tts_processor_instance = self.instance_tts_processor(self.voice_type, self.voice_name)
                while self.auto_speech_flag is True:  # user holds down the space bar
                    keyboard.add_hotkey('ctrl+shift', self.auto_speech_set, args=(True,))
                    keyboard.add_hotkey('ctrl+alt', self.chunk_speech, args=(True,))
                    keyboard.add_hotkey('shift+alt', self.interrupt_speech)
                    try:
                        # Record audio from microphone
                        audio = self.get_audio()
                        if self.listen_flag is True:
                            # Recognize speech to text from audio
                            user_input_prompt = self.recognize_speech(audio)
                            print(self.colors["GREEN"] + f"<<< 👂 SPEECH RECOGNIZED 👂 >>> ") # + self.colors["BRIGHT_YELLOW"] + f"{user_input_prompt} " + self.colors["GREEN"] + "<<")
                            speech_done = True
                            self.chunk_flag = False
                            # print(f"CHUNK FLAG STATE: {self.chunk_flag}")
                            self.auto_speech_flag = False
                    except sr.UnknownValueError:
                        #TODO REPLACE GOOGLE WITH WHISPER MODEL
                        print(self.colors["OKCYAN"] + "Google Speech Recognition could not understand audio" + self.colors["OKCYAN"])
                    except sr.RequestError as e:
                        print(self.colors["OKCYAN"] + "Could not request results from Google Speech Recognition service; {0}".format(e) + self.colors["OKCYAN"])
            elif self.listen_flag is False:
                user_input_prompt = input(self.colors["GREEN"] + f"<<< 🧠 USER 🧠 >>> " + self.colors["END"])
                speech_done = True
            user_input_prompt = self.voice_command_select_filter(user_input_prompt)
            cmd_run_flag = self.command_select(user_input_prompt)
            # get screenshot
            if self.llava_flag is True:
                self.screen_shot_flag = self.screen_shot_collector_instance.get_screenshot()
            # splice videos
            if self.splice_flag == True:
                self.data_set_video_process_instance.generate_image_data()
            if cmd_run_flag == False and speech_done == True:
                print(self.colors["YELLOW"] + f"{user_input_prompt}" + self.colors["OKCYAN"])
                # Send the prompt to the assistant
                if self.screen_shot_flag is True:
                    response = self.send_prompt(user_input_prompt)
                    self.screen_shot_flag = False
                else:
                    response = self.send_prompt(user_input_prompt)
                print(self.colors["RED"] + f"<<< 🤖 {self.user_input_model_select} 🤖 >>> " + self.colors["BRIGHT_BLACK"] + f"{response}" + self.colors["RED"])
                # Check for latex and add to queue
                if self.latex_flag:
                    # Create a new instance
                    latex_render_instance = latex_render_class()
                    latex_render_instance.add_latex_code(response, self.user_input_model_select)
                # Preprocess for text to speech, add flag for if text to speech enable handle canche otherwise do /leap or smt
                # Clear speech cache and split the response into sentences for next TTS cache
                #TODO alternative option for shut up fetature?
                # if self.leap_flag is not None and not self.leap_flag:
                #     self.tts_processor_instance.process_tts_responses(response, self.tts_processor_instance.voice_name)
                
                if self.leap_flag is not None and isinstance(self.leap_flag, bool):
                    if self.leap_flag != True:
                        self.tts_processor_instance.process_tts_responses(response, self.voice_name)
                        #TODO COMPLETE SHUTUP FEATURE INTERRUPT
                        if self.speech_interrupted:  # Add this check
                            print("Speech was interrupted. Ready for next input.")
                            self.speech_interrupted = False
                elif self.leap_flag is None:
                    pass
                # Start the mainloop in the main thread
                # print(self.colors["GREEN"] + f"<<< 🧠 USER 🧠 >>> " + self.colors["END"])

    # -------------------------------------------------------------------------------------------------        
    def interrupt_speech(self):
        self.speech_interrupted = True
        if hasattr(self, 'tts_processor_instance'):
            sd.stop()  # Stop any currently playing audio

    # -------------------------------------------------------------------------------------------------
    def generate_synthetic_data(self):
        """
        Generates synthetic data based on the given dataset and model prompt function.

        Returns:
            datasets.Dataset: A new dataset containing synthetic data.
        """
        # Prompt user for the old dataset name
        old_dataset_name = input("Enter the name of the old dataset: ")
        dataset_lib_dir = os.path.join(self.model_git_dir, "Finetune_Datasets")

        # Prompt user for the new dataset name
        new_dataset_name = input("Enter the name for the new dataset: ")

        # Construct the old dataset directory path
        old_dataset_dir = os.path.join(dataset_lib_dir, old_dataset_name, "data")

        # Create a new directory for the new dataset
        new_dataset_dir = os.path.join(dataset_lib_dir, new_dataset_name, "data")
        os.makedirs(new_dataset_dir, exist_ok=True)

        synthetic_data = []  # Initialize an empty list to store synthetic examples

        # Process Parquet files in the old dataset directory
        for parquet_file in os.listdir(old_dataset_dir):
            if parquet_file.endswith('.parquet'):
                parquet_path = os.path.join(old_dataset_dir, parquet_file)
                # Read the first data point from the Parquet file
                first_data_point = self.read_first_data_point(parquet_path)
                # Construct a prompt for each example
                print(f"here is the first data point: {first_data_point}")
                
                construct_prompt = f"Please generate 10 alternative variations in phrasing and structure for the following Hugging Face data point for a user assistant conversation training set: {first_data_point}"
                synthetic_example = self.shot_prompt(construct_prompt)  # Replace with your model call
                synthetic_data.append({'text': synthetic_example})

        # Create a new dataset from the synthetic data
        synthetic_dataset = Dataset.from_dict({'text': [item['text'] for item in synthetic_data]})

        # Write the synthetic dataset to a Parquet file
        synthetic_parquet_file = os.path.join(new_dataset_dir, "synthetic_dataset.parquet")
        synthetic_dataset.to_pandas().to_parquet(synthetic_parquet_file)

        return synthetic_dataset

    
    # -------------------------------------------------------------------------------------------------
    def read_first_data_point(self, parquet_path):
        # Read the Parquet file and extract the first data point
        table = pq.read_table(parquet_path)
        first_data_point = table.to_pandas().iloc[0]
        return first_data_point
