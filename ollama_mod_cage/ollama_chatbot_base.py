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

    By: Leo Borcherding, 4/20/2024
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
import threading
import json
import asyncio
import numpy as np

from wizard_spell_book.Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands
from wizard_spell_book.Public_Chatbot_Base_Wand.speech_to_speech import tts_processor_class
from wizard_spell_book.Public_Chatbot_Base_Wand.directory_manager import directory_manager_class
from wizard_spell_book.Public_Chatbot_Base_Wand.latex_render import latex_render_class
from wizard_spell_book.Public_Chatbot_Base_Wand.data_set_manipulator import data_set_constructor
from wizard_spell_book.Public_Chatbot_Base_Wand.write_modelfile import model_write_class
from wizard_spell_book.Public_Chatbot_Base_Wand.read_write_symbol_collector import read_write_symbol_collector
from wizard_spell_book.Public_Chatbot_Base_Wand.data_set_manipulator import screen_shot_collector
from wizard_spell_book.Public_Chatbot_Base_Wand.create_convert_model import create_convert_manager
from wizard_spell_book.Public_Chatbot_Base_Wand.node_custom_methods import FileSharingNode
from wizard_spell_book.Public_Chatbot_Base_Wand.speech_to_speech import speech_recognizer_class

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
import sounddevice as sd
import speech_recognition as sr

# TODO setup sebdg emotional classifyer keras 
# from tensorflow.keras.models import load_model
# sentiment_model = load_model('D:\\CodingGit_StorageHDD\\model_git\\emotions_classifier\\emotions_classifier.keras')

# -------------------------------------------------------------------------------------------------
class ollama_chatbot_base:
    """ 
    This class provides an interface to the Ollama local serve API for creating custom chatbot agents.
    It also provides access to Speech-to-Text transcription and Text-to-Speech generation methods via a low-level command line interface and the coqui-TTS, XTTS model.
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
        # self.set_model()
        # self.user_input_model_select = self.user_input_model_select

        # initialize chat
        self.chat_history = []
        self.llava_history = []
        self.agent_library = []
        self.agent_dict = []
        
        # initialize prompt args
        self.LLM_SYSTEM_PROMPT_FLAG = False
        self.LLM_BOOSTER_PROMPT = False
        self.VISION_SYSTEM_PROMPT = False
        self.VISION_BOOSTER_PROMPT = False

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

        #TODO ADD screen shot {clock & manager}
        self.screenshot_path = os.path.join(self.llava_library, "screenshot.png")
        
        # build conversation save path #TODO ADD TO DEV DICT
        self.default_conversation_path = os.path.join(self.parent_dir, f"AgentFiles\\Ignored_pipeline\\conversation_library\\{self.user_input_model_select}\\{self.save_name}.json")

        # text section
        self.latex_flag = False
        self.cmd_run_flag = None

        # agent select flag
        self.agent_flag = False
        self.memory_clear = False
        
        # ollama chatbot base setup wand class instantiation
        self.ollama_command_instance = ollama_commands(self.user_input_model_select, self.developer_tools_dict)
        self.colors = self.ollama_command_instance.colors
        
        # initialize speech_recognizer_class
        self.speech_recognizer_instance = speech_recognizer_class(self.colors)
        self.speech_interrupted = False
        
        self.hotkeys = {
            'ctrl+shift': self.start_speech_recognition,
            'ctrl+alt': self.stop_speech_recognition,
            'shift+alt': self.interrupt_speech,
        }
        self.audio_data = np.array([])
        self.speech_recognition_active = False

        # speech flags:
        self.leap_flag = True # TODO turn off for minecraft
        self.listen_flag = False # TODO turn off for minecraft
        self.chunk_flag = False
        self.auto_speech_flag = False #TODO keep off BY DEFAULT FOR MINECRAFT, TURN ON TO START

        # vision flags:
        self.llava_flag = False # TODO TURN ON FOR MINECRAFT
        self.splice_flag = False
        self.screen_shot_flag = False

        # get directory data
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
    def set_model(self, model_name):
        self.user_input_model_select = model_name
        print(f"Model set to {model_name}")
        
    # ------------------------------------------------------------------------------------------------- 
    def swap(self):
        self.chat_history = []
        print(f"Model changed to {self.user_input_model_select}")
        return

    # ------------------------------------------------------------------------------------------------- 
    def set_voice(self, voice_type, voice_name):
        self.voice_type = voice_type
        self.voice_name = voice_name
        self.tts_processor_instance = self.instance_tts_processor(voice_type, voice_name)
        print(f"Voice set to {voice_name} (Type: {voice_type})")

    # ------------------------------------------------------------------------------------------------- 
    async def get_available_voices(self):
        fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
        fine_tuned_voices = [d[8:] for d in os.listdir(fine_tuned_dir) if os.path.isdir(os.path.join(fine_tuned_dir, d)) and d.startswith("XTTS-v2_")]
        reference_voices = [d for d in os.listdir(self.tts_voice_ref_wav_pack_path) if os.path.isdir(os.path.join(self.tts_voice_ref_wav_pack_path, d))]
        return {"fine_tuned": fine_tuned_voices, "reference": reference_voices}
    
    # ------------------------------------------------------------------------------------------------- 
    
    def set_speech(self, enabled: bool):
        self.speech_enabled = enabled
        return f"Speech {'enabled' if enabled else 'disabled'}"
            
    # ------------------------------------------------------------------------------------------------- 
    def setup_hotkeys(self):
        for hotkey, callback in self.hotkeys.items():
            keyboard.add_hotkey(hotkey, callback)
            
    # ------------------------------------------------------------------------------------------------- 
    def remove_hotkeys(self):
        for hotkey in self.hotkeys:
            keyboard.remove_hotkey(hotkey)

    # -------------------------------------------------------------------------------------------------
    def start_speech_recognition(self):
        self.speech_recognition_active = True
        # Start recording audio
        self.start_audio_stream()
        
    # -------------------------------------------------------------------------------------------------
    def stop_speech_recognition(self):
        self.speech_recognition_active = False
        # Stop recording audio and process the recorded speech
        self.stop_audio_stream()
        recognized_text = self.process_speech()
        return recognized_text

    # -------------------------------------------------------------------------------------------------
    def interrupt_speech(self):
        if hasattr(self, 'tts_processor_instance'):
            self.tts_processor_instance.interrupt_generation()

    # -------------------------------------------------------------------------------------------------
    def start_audio_stream(self):
        self.audio_data = np.array([])
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=44100,
                                  input=True,
                                  frames_per_buffer=1024,
                                  stream_callback=self.audio_callback)
        self.stream.start_stream()

    # -------------------------------------------------------------------------------------------------
    def stop_audio_stream(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()

    # -------------------------------------------------------------------------------------------------
    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_data = np.concatenate((self.audio_data, audio_data))
        return (in_data, pyaudio.paContinue)
    
    # -------------------------------------------------------------------------------------------------
    def get_user_audio_data(self):
        return self.speech_recognizer_instance.get_audio_data()
    
    # -------------------------------------------------------------------------------------------------
    def get_llm_audio_data(self):
        return self.tts_processor_instance.get_audio_data()
    
    # # -------------------------------------------------------------------------------------------------  
    # def get_model(self):
    #     """ a method for collecting the model name from the user input
    #     """
    #     HEADER = '\033[95m'
    #     OKBLUE = '\033[94m'
    #     self.user_input_model_select = input(HEADER + "<<< PROVIDE MODEL NAME >>> " + OKBLUE)
    
    # # -------------------------------------------------------------------------------------------------
    # def swap(self):
    #     """ a method to call when swapping models
    #     """
    #     self.chat_history = []
    #     self.user_input_model_select = input(self.colors['HEADER']+ "<<< PROVIDE AGENT NAME TO SWAP >>> " + self.colors['OKBLUE'])
    #     print(f"Model changed to {self.user_input_model_select}")
    #     return
    
    # ------------------------------------------------------------------------------------------------
    def get_vision_model(self):
            #TODO LOAD LLAVA AND PHI3 CONCURRENTLY, prompt phi, then send to llava, but dont instance llava when prompting
            self.vision_model_select = input(self.colors["HEADER"] + "<<< PROVIDE VISION MODEL NAME >>> " + self.colors["OKBLUE"])
            #TODO ADD BETTER VISION MODEL SELECTOR
            if self.vision_model_select is None:
                print("NO MODEL SELECTED, DEFAULTING TO llava")
                vision_model = "llava"
            else:
                vision_model = self.vision_model_select
            self.vision_model = vision_model

    # # -------------------------------------------------------------------------------------------------   
    # def get_audio(self):
    #     """ a method for getting the user audio from the microphone
    #         args: none
    #     """
    #     print(">>AUDIO RECORDING<<")
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    #     frames = []

    #     while self.auto_speech_flag and not self.chunk_flag:
    #         data = stream.read(1024)
    #         frames.append(data)

    #     print(">>AUDIO RECEIVED<<")
    #     stream.stop_stream()
    #     stream.close()
    #     p.terminate()

    #     # Convert the audio data to an AudioData object
    #     audio = sr.AudioData(b''.join(frames), 16000, 2)
    #     self.chunk_flag = False  # Set chunk_flag to False here to indicate that the audio has been received
    #     return audio

    # -------------------------------------------------------------------------------------------------
    def get_system_audio(self):
        """ a method to get the system audio from discord, spotify, youtube, etc, to be recognized by
            the speech to text model
            args: none
            returns: none
        """
        test = "test"
        #TODO grab audio clip, and transcibe to text for input, also add transcibe and save to text json
        # also add get audio and store as audio for training and add record clone for user mic to train xtts 

    # # -------------------------------------------------------------------------------------------------   
    # def recognize_speech(self, audio):
    #     """ a method for calling the speech recognizer
    #         args: audio
    #         returns: speech_str
    #     """
    #     #TODO Realized current implementation calls google API, must replace with LOCAL SPEECH RECOGNITION MODEL
    #     speech_str = sr.Recognizer().recognize_google(audio)
    #     print(f">>{speech_str}<<")
    #     return speech_str
    
    # # ------------------------------------------------------------------------------------------------- 
    # def wake_words(self, audio):
    #     """ a method for recognizing speech with the wake commands.
    #     """
    #     speech_str = sr.Recognizer().recognize_google(audio)
    #     wake_word = "Hey Assistant"
    #     if speech_str.lower().startswith(wake_word.lower()):
    #         # Remove the wake word from the speech string
    #         actual_command = speech_str[len(wake_word):].strip()
    #         return actual_command
    #     return None

    # -------------------------------------------------------------------------------------------------- 
    def navigator_default(self):
        """ a method to get the default vision navigator prompt
        """
        # general_navigator
        self.general_navigator_agent = {
            "LLM_SYSTEM_PROMPT" : (
                "You are a helpful llm assistant, designated with with fulling the user's request, "
                "the user is communicating with speech recognition and is sending their "
                "screenshot data to the vision model for decomposition. Receive this destription and "
                "Instruct the user and help them fullfill their request by collecting the vision data "
                "and responding. "
            ), 
            "LLM_BOOSTER_PROMPT" : (
                "Here is the output from the vision model describing the user screenshot data "
                "along with the users speech data. Please reformat this data, and formulate a "
                "fullfillment for the user request in a conversational speech manner which will "
                "be processes by the text to speech model for output. "
            ),
            "VISION_SYSTEM_PROMPT" : (
                "You are an image recognition assistant, the user is sending you a request and an image "
                "please fullfill the request"
            ), 
            "VISION_BOOSTER_PROMPT" : (
                "Given the provided screenshot, please provide a list of objects in the image "
                "with the attributes that you can recognize. "
            )
        }
        return
    
    # --------------------------------------------------------------------------------------------------
    def agent_prompt_library(self):
        """ a method to setup the agent prompt dictionaries for the user
            #TODO add agent prompt collection from .modelfile or .agentfile
            args: none
            returns: none

            =-=-=-= 👽 =-=-=-= AGENT PROMPT LIBRARY =-=-=-= 👽 =-=-=-=

            🤖 system prompt 🤖
                self.chat_history.append({"role": "system", "content": "selected_system_prompt"})

            🧠 user prompt booster 🧠 
                self.chat_history.append({"role": "user", "content": "selected_booster_prompt"}) 

            =-=-=-= 🔥 =-=-=-= AGENT  STRUCTURE =-=-=-= 🌊 =-=-=-=

            self.agent_name = {
                "agent_llm_system_prompt" : (
                    "You are a helpful llm assistant, designated with with fulling the user's request, "
                ), 
                "agent_llm_booster_prompt" : (
                    "Here is the llava data/latex data/etc from the vision/latex/rag action space, "
                ),
                "agent_vision_system_prompt" : (
                    "You are a multimodal vision to text model, the user is navigating their pc and "
                    "they need a description of the screen, which is going to be processed by the llm "
                    "and sent to the user after this. please return of json output of the image data. "
                ), 
                "agent_vision_booster_prompt" : (
                    "You are a helpful llm assistant, designated with with fulling the user's request, "
                ), 
        """
        #TODO ADD WRITE AGENTFILE
        #TODO add text prompts for the following ideas:
        # latex pdf book library rag
        # c3po adventure
        # rick and morty adveture
        # phi3 & llama3 fast shot prompting 
        # linked in, redbubble, oarc - advertising server api for laptop
        #TODO add create .agentfile with voice commands or text input
        #CSV CONSTRUCTION
        #TODO FUNCTION CALLING MODEL DATASET
        #TODO LATEX MATH MODEL DATASET
        #LABEL STUDIO
        #TODO MINECRAFT LLAVA DATASET
        #TODO WINDOWS LLAVA DATASET
        #TODO NAMECALL, SMART LISTEN, FAQ CHECKER
        #TODO DUCK DUCK GO SEARCH
        #TODO LATEX MODEL -> MATPLOT LIVE AI GRAPH CALC
        #TODO ROUTING FOR FUNTION CALLING MODEL TO EXECUTE FUNCTION CALLS
        #TODO SIMPLE LANGRAPH RAG
        #TODO SIMPLE USEFUL KERAS MODEL EITHER EMOTIONS OR SOMETHING ELSE
        #TODO ADD DUCKDUCKGO SEARCH PROMPT BOOST
        #TODO RECORD VOICE MEMO/CLONE DATA
        #TODO PLAY MUSIC, MOVIE, MP3, MP4 from library
        #TODO BOOK AUDIO, EITHER FROM TEXT OR AUDIO FILE
        #TODO LATEX PDF GENERATE FILE
        #TODO CSV DATA SYNTHESIS
        #TODO COMFYUI Image generation workflows
        #TODO text to image
        # img to video
        # img to img
        # uncrop image extender
        # new stable diffusion 3 models, img gen model library
        # lora select, workflow select
        # TODO FROG ANIMATED PORTRAIT
        # TODO LIP SYNC AVATAR LIVE
        # TODO WEBCAM LLAVA RECOGNIZER, objects, emotions, math, text, art, nature
        #TODO SMART LISTEN 1 -> listens afer long pause outputs /moderator, faq checker
        #TODO yo llama pull that up

        #TODO preload command list -> command.txt run desired default command setup with /preload command list [name]
        #TODO macro jobset for keyboard and mouse input for the entire program
        #TODO UPDATE/UPGRADE COMPREHENSIVE SENTENCE PARSER HANDLE BULLET POINTS, NUMBERS, "", code, emojis, and more, just something
        #TODO ^-- if great than 250 with no splice, then splice at the closest word under 250
        #TODO german, spanish, french, english whisper models
        #TODO that new ollama 0.2.1 multilingual model
        #TODO DESTINY DINKLEBOT AGENT
        #TODO MTG ASSIST AGENT
        #TODO YOUTUBE REVIEW AGENT
        #TODO PEER TO PEER NETWORK
        #TODO EMAIL SHARING AGENT
        #TODO CREW AI?
        #TODO UNSLOTH AUTOMATIONS
        #TODO LLAMACPP, GGUF AND SAFETENSOR CONVERT UPGRADE

        #TODO MAIN TODO's BEFORE AI MAKERSPACE SESSION

        # --------------------------------------------------------------------------------------------------
        # minecraft
        self.minecraft_agent = {
            "LLM_SYSTEM_PROMPT" : (
                "You are a helpful Minecraft assistant. Given the provided screenshot data, "
                "please direct the user immediately. Prioritize the order in which to inform "
                "the player. Hostile mobs should be avoided or terminated. Danger is a top "
                "priority, but so is crafting and building. If they require help, quickly "
                "guide them to a solution in real time. Please respond in a quick conversational "
                "voice. Do not read off documentation; you need to directly explain quickly and "
                "effectively what's happening. For example, if there is a zombie, say something "
                "like, 'Watch out, that's a Zombie! Hurry up and kill it or run away; they are "
                "dangerous.' The recognized objects around the perimeter are usually items, health, "
                "hunger, breath, GUI elements, or status effects. Please differentiate these objects "
                "in the list from 3D objects in the forward-facing perspective (hills, trees, mobs, etc.). "
                "The items are held by the player and, due to the perspective, take up the warped edge "
                "of the image on the sides. The sky is typically up with a sun or moon and stars, with "
                "the dirt below. There is also the Nether, which is a fiery wasteland, and cave systems "
                "with ore. Please stick to what's relevant to the current user prompt and lava data."
            ),
            "LLM_BOOSTER_PROMPT" : (
                "Based on the information in LLAVA_DATA please direct the user immediatedly, prioritize the "
                "order in which to inform the player of the identified objects, items, hills, trees and passive "
                "and hostile mobs etc. Do not output the dictionary list, instead conversationally express what "
                "the player needs to do quickly so that they can ask you more questions."
            ),
            "VISION_SYSTEM_PROMPT": (
                "You are a Minecraft image recognizer assistant. Search for passive and hostile mobs, "
                "trees and plants, hills, blocks, and items. Given the provided screenshot, please "
                "provide a dictionary of the recognized objects paired with key attributes about each "
                "object, and only 1 sentence to describe anything else that is not captured by the "
                "dictionary. Do not use more sentences. Objects around the perimeter are usually player-held "
                "items like swords or food, GUI elements like items, health, hunger, breath, or status "
                "affects. Please differentiate these objects in the list from the 3D landscape objects "
                "in the forward-facing perspective. The items are held by the player traversing the world "
                "and can place and remove blocks. Return a dictionary and 1 summary sentence."
            ),
            "VISION_BOOSTER_PROMPT": (
                "Given the provided screenshot, please provide a dictionary of key-value pairs for each "
                "object in the image with its relative position. Do not use sentences. If you cannot "
                "recognize the enemy, describe the color and shape as an enemy in the dictionary, and "
                "notify the LLMs that the user needs to be warned about zombies and other evil creatures."
            )
        }

        # --------------------------------------------------------------------------------------------------
        # general_navigator
        self.general_navigator_agent = {
            "LLM_SYSTEM_PROMPT" : (
                "You are a helpful llm assistant, designated with with fulling the user's request, "
                "the user is communicating with speech recognition and is sending their "
                "screenshot data to the vision model for decomposition. Receive this destription and "
                "Instruct the user and help them fullfill their request by collecting the vision data "
                "and responding. "
            ), 
            "LLM_BOOSTER_PROMPT" : (
                "Here is the output from the vision model describing the user screenshot data "
                "along with the users speech data. Please reformat this data, and formulate a "
                "fullfillment for the user request in a conversational speech manner which will "
                "be processes by the text to speech model for output. "
            ),
            "VISION_SYSTEM_PROMPT" : (
                "You are an image recognition assistant, the user is sending you a request and an image "
                "please fullfill the request"
            ), 
            "VISION_BOOSTER_PROMPT" : (
                "Given the provided screenshot, please provide a list of objects in the image "
                "with the attributes that you can recognize. "
            )
        }
        # --------------------------------------------------------------------------------------------------
        # phi3_speed_chat
        self.phi3_speed_chat_agent = {
            "LLM_SYSTEM_PROMPT" : 
                "You are borch/phi3_speed_chat, a phi3 large language model, specifically you have been "
                "tuned to respond in a more quick and conversational manner, the user is using speech to "
                "text for communication, its also okay to be fun and wild as a phi3 ai assistant. Its also "
                "okay to respond with a question, if directed to do something just do it, and realize that "
                "not everything needs to be said in one shot, have a back and forth listening to the users "
                "response. If the user decides to request a latex math code output, use \[...\] instead of "
                "$$...$$ notation, if the user does not request latex, refrain from using latex unless "
                "necessary. Do not re-explain your response in a parend or bracketed note: the response... "
                "this is annoying and users dont like it.",
        }

    # -------------------------------------------------------------------------------------------------
    def agent_prompt_select(self):
        """ a method for displaying the agent library and requesting the user to select the agent to load
        """  
        self.agent_flag = True
        # print(self.colors["LIGHT_MAGENTA"] + "<<< AGENT PROMPTS >>>")

        # # for agents in library print the agent name
        # for agents in self.agent_library:
        #     print(self.colors["OKBLUE"] + f"AGENT NAME:" + self.colors["LIGHT_YELLOW"] + f" {agents}")
        #     print(self.colors["OKBLUE"] + f"AGENT NAME:" + self.colors["LIGHT_YELLOW"] + f" {self.agent_library[agents]}")
        #     for items in agents:
        #         # for items in the agent print the items
        #         print(self.colors["OKBLUE"] + f"AGENT ITEMS:" + self.colors["RED"] + f" {items}")
        #         print(self.colors["OKBLUE"] + f"AGENT ITEMS:" + self.colors["RED"] + f" {self.agents[items]}")
        #         print(self.colors["OKBLUE"] + f"AGENT ITEMS:" + self.colors["RED"] + f" {self.agent_library[agents][items]}")

        self.agent_prompt_library()

        print(self.colors["BRIGHT_YELLOW"] + "<<< AGENT LIBRARY >>> ")

        print(self.colors["OKBLUE"] + "<<< minecraft_agent >>> ")
        for item in self.minecraft_agent:
            print(self.colors["BRIGHT_YELLOW"] + f"<<< {item} >>> " + self.colors["RED"] + f"{self.minecraft_agent[item]}")

        print(self.colors["OKBLUE"] + "<<< general_navigator_agent >>> ")
        for item in self.general_navigator_agent:
            print(self.colors["BRIGHT_YELLOW"] + f"<<< {item} >>> " + self.colors["RED"] + f"{self.general_navigator_agent[item]}")

        print(self.colors["OKBLUE"] + "<<< phi3_speed_chat_agent >>> ")
        for item in self.phi3_speed_chat_agent:
            print(self.colors["BRIGHT_YELLOW"] + f"<<< {item} >>> " + self.colors["RED"] + f"{self.phi3_speed_chat_agent[item]}")

        self.agent_select = input(self.colors["HEADER"] + "<<< PROVIDE AGENT SYSTEM PROMPT NAME >>> " + self.colors["OKBLUE"])

        if self.agent_select == "minecraft_agent":
            self.agent_dict = self.minecraft_agent
            self.LLM_SYSTEM_PROMPT_FLAG = True
            self.LLM_BOOSTER_PROMPT = True
            self.VISION_SYSTEM_PROMPT = True
            self.VISION_BOOSTER_PROMPT = True

        if self.agent_select == "general_navigator_agent":
            self.agent_dict = self.general_navigator_agent
            self.LLM_SYSTEM_PROMPT_FLAG = True
            self.LLM_BOOSTER_PROMPT = True
            self.VISION_SYSTEM_PROMPT = True
            self.VISION_BOOSTER_PROMPT = True

        if self.agent_select == "phi3_speed_chat_agent":
            self.agent_dict = self.phi3_speed_chat_agent
            self.LLM_SYSTEM_PROMPT_FLAG = True
            self.LLM_BOOSTER_PROMPT = False
            self.VISION_SYSTEM_PROMPT = False
            self.VISION_BOOSTER_PROMPT = False
            
    # -------------------------------------------------------------------------------------------------
    def shot_prompt(self, prompt):
        # Clear chat history
        self.shot_history = []

        # Append user prompt
        self.shot_history.append({"role": "user", "content": prompt})

        try:
            response = ollama.generate(model=self.user_input_model_select, prompt=prompt, stream=True)
            
            model_response = ''
            for chunk in response:
                if 'response' in chunk:
                    content = chunk['response']
                    model_response += content
                    print(content, end='', flush=True)
            
            print('\n')
            
            # Append the full response to shot_history
            self.shot_history.append({"role": "assistant", "content": model_response})
            
            return model_response
        except Exception as e:
            return f"Error: {e}"
        
    # -------------------------------------------------------------------------------------------------   
    def send_prompt(self, user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        # if agent selected set up system prompts for llm
        if self.agent_flag == True:
            self.chat_history.append({"role": "system", "content": self.agent_dict["LLM_SYSTEM_PROMPT"]})
            print(self.colors["OKBLUE"] + f"<<< LLM SYSTEM >>> ")
            print(self.colors["BRIGHT_YELLOW"] + f"{self.agent_dict['LLM_SYSTEM_PROMPT']}")
        else:
            pass

        #TODO ADD IF MEM OFF CLEAR HISTORY, also add long term memory support with rag and long term conversation file for demo
        if self.memory_clear == True:
            self.chat_history = []

        # append user prompt from text or speech input
        self.chat_history.append({"role": "user", "content": user_input_prompt})

        # get the llava response and append it to the chat history only if an image is provided
        if self.llava_flag is True:
            # load the screenshot and convert it to a base64 string
            with open(f'{self.screenshot_path}', 'rb') as f:
                user_screenshot_raw2 = base64.b64encode(f.read()).decode('utf-8')
                self.user_screenshot_raw = user_screenshot_raw2

            # get llava response from constructed user input
            self.llava_response = self.llava_prompt(user_input_prompt, user_screenshot_raw2, user_input_prompt, self.vision_model)

            # if agent selected set up intermediate prompts for llm model
            if self.LLM_BOOSTER_PROMPT == True:
                self.chat_history.append({"role": "assistant", "content": f"VISION_DATA: {self.llava_response}"})
                self.llm_booster_constructor = self.agent_dict["LLM_BOOSTER_PROMPT"] + f"{user_input_prompt}"
                self.chat_history.append({"role": "user", "content": self.llm_booster_constructor})
                print(self.colors["OKBLUE"] + f"<<< LLM BOOSTER >>> ")
                print(self.colors["BRIGHT_YELLOW"] + f"<<< {self.llm_booster_constructor} >>>")
            else:
                self.navigator_default()
                self.chat_history.append({"role": "assistant", "content": f"VISION_DATA: {self.llava_response}"})
                self.llm_booster_constructor = self.general_navigator_agent["LLM_BOOSTER_PROMPT"] + f"{user_input_prompt}"
                self.chat_history.append({"role": "user", "content": self.llm_booster_constructor})
                print(self.colors["OKBLUE"] + f"<<< LLM BOOSTER >>> ")
                print(self.colors["BRIGHT_YELLOW"] + f"<<< {self.llm_booster_constructor} >>>")

        try:
            # Send user input or user input & llava output to the selected LLM
            response = ollama.chat(model=self.user_input_model_select, messages=self.chat_history, stream=True)
            
            model_response = ''
            print(self.colors["RED"] + f"<<< 🤖 {self.user_input_model_select} 🤖 >>> " + self.colors["BRIGHT_BLACK"], end='', flush=True)
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    model_response += content
                    print(content, end='', flush=True)
            
            print(self.colors["RED"])  # Reset color after the response
            
            # Append the full response to chat history
            self.chat_history.append({"role": "assistant", "content": model_response})
            
            # Process the response with the text-to-speech processor
            if self.leap_flag is not None and isinstance(self.leap_flag, bool):
                if not self.leap_flag:
                    self.tts_processor_instance.process_tts_responses(model_response, self.voice_name)
                    if self.speech_interrupted:
                        print("Speech was interrupted. Ready for next input.")
                        self.speech_interrupted = False
            
            return model_response
        except Exception as e:
            return f"Error: {e}"
        
    # -------------------------------------------------------------------------------------------------   
    def llava_prompt(self, user_input_prompt, user_screenshot_raw2, llava_user_input_prompt, vision_model="llava"):
        """ a method for prompting the vision model
            args: user_screenshot_raw2, llava_user_input_prompt, vision_model="llava"
            returns: none

            #TODO default? if none selected?
            #TODO add modelfile, system prompt get feature and modelfile manager library
            #TODO /sys prompt select, /booster prompt select, ---> leverage into function calling ai 
            for modular auto prompting chatbot
        """ 
        # setup history & prompt
        self.llava_user_input_prompt = llava_user_input_prompt
        self.llava_history = []

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.llava_history.append({"role": "system", "content": "You are a minecraft image recognizer assistant, 
        # search for passive and hostile mobs, trees and plants, hills, blocks, and items, given the provided screenshot 
        # in the forward facing perspective, the items are held by the player traversing the world and can place and remove blocks. 
        # Return dictionary and 1 summary sentence:"})
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # if agent selected, set up system prompts for vision model

        if self.VISION_SYSTEM_PROMPT is True:
            self.vision_system_constructor = f"{self.agent_dict['VISION_SYSTEM_PROMPT']} " + f"{user_input_prompt}"
            self.llava_history.append({"role": "system", "content": f"{self.vision_system_constructor}"})
            print(f"<<< VISION SYSTEM >>> " + self.colors['BRIGHT_YELLOW'] + f"{self.vision_system_constructor} >>>")
        else:
            self.navigator_default()
            self.vision_system_constructor = f"{self.general_navigator_agent['VISION_SYSTEM_PROMPT']} " + f"{user_input_prompt}"
            self.llava_history.append({"role": "system", "content": f"{self.vision_system_constructor}"})
            print(f"<<< VISION SYSTEM >>> " + self.colors['BRIGHT_YELLOW'] + f"{self.vision_system_constructor} >>>")

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # message = {"role": "user", 
        #            "content": "given the provided screenshot please provide a dictionary of key value pairs for each object in " 
        #            "with image with its relative position, do not use sentences, if you cannot recognize the enemy describe the " 
        #            "color and shape as an enemy in the dictionary, and notify the llms that the user needs to be warned about "
        #            "zombies and other evil creatures"}
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if self.agent_flag == True:
            self.vision_booster_constructor = f"{self.agent_dict['VISION_BOOSTER_PROMPT']}" + f"{user_input_prompt}"
            message = {"role": "user", "content": f"{self.vision_booster_constructor}"}
            print(self.colors["OKBLUE"] + f"<<< VISION BOOSTER >>> " + self.colors['BRIGHT_YELLOW'] + f"{self.vision_booster_constructor} >>>")
        else:
            self.vision_booster_constructor = f"{self.general_navigator_agent['VISION_BOOSTER_PROMPT']}" + f"{user_input_prompt}"
            message = {"role": "user", "content": f"{self.vision_booster_constructor}"}
            print(self.colors["OKBLUE"] + f"<<< VISION BOOSTER >>> " + self.colors['BRIGHT_YELLOW'] + f"{self.vision_booster_constructor} >>>")

        #TODO ADD LLM PROMPT REFINEMENT (example: stable diffusion prompt model) AS A PREPROCESS COMBINED WITH THE CURRENT AGENTS PRIME DIRECTIVE
        if user_screenshot_raw2 is not None:
            # Assuming user_input_image is a base64 encoded image
            message["images"] = [user_screenshot_raw2]
        try:
            # Prompt vision model with compiled chat history data
            response_llava = ollama.chat(model=vision_model, messages=self.llava_history + [message], stream=True)
            
            model_response = ''
            print(self.colors["RED"] + f"<<< 🖼️ {vision_model} 🖼️ >>> " + self.colors["BRIGHT_BLACK"], end='', flush=True)
            for chunk in response_llava:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    model_response += content
                    print(content, end='', flush=True)
            
            print(self.colors["RED"])  # Reset color after the response
            
            # Append the full response to llava_history
            self.llava_history.append({"role": "assistant", "content": model_response})
            
            # Keep only the last 2 responses in llava_history
            self.llava_history = self.llava_history[-2:]

            return model_response
        except Exception as e:
            return f"Error: {e}"
        
    # -------------------------------------------------------------------------------------------------
    def voice_command_select_filter(self, user_input_prompt):
        """ a method for managing the voice command selection
            Args: user_input_prompt
            Returns: user_input_prompt
        """ 
        # Parse for general commands (non token specific args)
        #TODO ADD NEW FUNCTIONS and make llama -> ollama lava -> llava, etc
        user_input_prompt = re.sub(r"activate swap", "/swap", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate quit", "/quit", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate llama create", "/ollama create", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate llama show", "/ollama show", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate llama template", "/ollama template", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate llama license", "/ollama license", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate llama list", "/ollama list", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate llama loaded", "/ollama loaded", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate listen on", "/listen on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate listen off", "/listen off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate speech on", "/speech on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate speech off", "/speech off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate voice off", "/voice off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate voice on", "/voice on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate latex on", "/latex on", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate latex off", "/latex off", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate lava flow", "/llava flow", user_input_prompt, flags=re.IGNORECASE)
        user_input_prompt = re.sub(r"activate lava freeze", "/llava freeze", user_input_prompt, flags=re.IGNORECASE)

        #TODO replace /llava flow/freeze with lava flow/freeze
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
            #TODO IMPLEMENT /system base prompt from .modelfile
        """
        self.command_library = {
            "/swap": lambda: self.swap(),
            "/agent select": lambda: self.agent_prompt_select(),
            "/voice swap": lambda: self.voice_swap(),
            "/save as": lambda: self.save_to_json(self.save_name, self.user_input_model_select),
            "/load as": lambda: self.load_from_json(self.load_name, self.user_input_model_select),
            "/write modelfile": lambda: self.model_write_class_instance.write_model_file(),
            "/convert tensor": lambda: self.create_convert_manager_instance.safe_tensor_gguf_convert(self.tensor_name),
            "/convert gguf": lambda: self.model_write_class_instance.write_model_file_and_run_agent_create_gguf(self.listen_flag, self.model_git),
            "/listen on": lambda: self.listen(),
            "/listen off": lambda: self.listen(),
            "/voice off": lambda: self.voice(True),
            "/voice on": lambda: self.voice(False),
            "/speech on": lambda: self.speech(False, True),
            "/speech off": lambda: self.speech(True, False),
            "/wake on" : lambda: self.wake_commands(True),
            "/wake off" : lambda: self.wake_commands(False),
            "/latex on": lambda: self.latex(True),
            "/latex off": lambda: self.latex(False),
            "/command auto on": lambda: self.auto_commands(True),
            "/command auto off": lambda: self.auto_commands(False),
            "/llava flow": lambda: self.llava_flow(True),
            "/llava freeze": lambda: self.llava_flow(False),
            "/auto on": lambda: self.auto_speech_set(True),
            "/auto off": lambda: self.auto_speech_set(False),
            "/quit": lambda: self.ollama_command_instance.quit(),
            "/ollama create": lambda: self.ollama_command_instance.ollama_create(),
            "/ollama show": lambda: self.ollama_command_instance.ollama_show_modelfile(),
            "/ollama template": lambda: self.ollama_command_instance.ollama_show_template(),
            "/ollama license": lambda: self.ollama_command_instance.ollama_show_license(),
            "/ollama list": lambda: self.ollama_command_instance.ollama_list(),
            "/ollama loaded": lambda: self.ollama_command_instance.ollama_show_loaded_models(),
            "/splice video": lambda: self.data_set_video_process_instance.generate_image_data(),
            "/developer new" : lambda: self.read_write_symbol_collector_instance.developer_tools_generate(),
            "/start node": lambda: self.FileSharingNode_instance.start_node(),
            "/synthetic generator": lambda: self.generate_synthetic_data(),
            "/convert wav": lambda: self.data_set_video_process_instance.call_convert()
        }
        
        # Find the command in the command string
        command = next((cmd for cmd in self.command_library.keys() if command_str.startswith(cmd)), None)

        # If a command is found, split it from the arguments
        if command:
            args = command_str[len(command):].strip()
        else:
            args = None

        # If Listen off
        if command == "/listen off":
            self.listen_flag = False
            self.speech_recognizer_instance.auto_speech_flag = False
            print("- speech to text deactivated -")
            return True  # Ensure this returns True to indicate the command was processed
        
        # If /llava flow, 
        if command == "/llava flow":
            # user select the vision model, #TODO SHOW OLLAMA LIST AND SUGGEST POSSIBLE VISION MODELS, ULTIMATLEY YOU NEED THE OLLAMA MODEL NAME
            self.get_vision_model()

        # If /system select, 
        if command == "/system select":
            # get the user selection for the system prompt, print system prompt name selector and return user input
            # self.get_system_prompts()
            self.agent_prompt_library()
            self.agent_prompt_select()

        # If /system select, 
        if command == "/system base":
            # get the system prompt library
            #self.TODO GET BASE SYSTEM PROMPT FROM /ollama modelfile
            test = None #DONT USE YET LOL 😂, JUST RESTART THE PROGRAM TO RETURN TO BASE

        # Check if the command is in the library, if not return None
        if command in self.command_library:
            #Call Command Method
            self.command_library[command]()

            cmd_run_flag = True
            return cmd_run_flag
        else:
            cmd_run_flag = False
            return cmd_run_flag
        
    # -------------------------------------------------------------------------------------------------
    def cmd_list(self):
        """ a method for printing the command list via /command list
        """
        print(self.colors['OKBLUE'] + "<<< COMMAND LIST >>>")
        for commands in self.command_library:
            print(self.colors['BRIGHT_YELLOW'] + f"{commands}")

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
        print(self.colors['RED'] + f"<<< Available voices >>>" + self.colors['BRIGHT_YELLOW'])
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
    # def chunk_speech(self, value):
    #     """
    #     This method sets the chunk_flag to the given value and prints its state.
    #     The chunk_flag is used to control whether the speech input should be chunked.

    #     Args:
    #         value (bool): The value to set the chunk_flag to.
    #     """
    #     # time.sleep(1)
    #     self.chunk_flag = value
    #     print(f"chunk_flag FLAG STATE: {self.chunk_flag}")

    # # -------------------------------------------------------------------------------------------------   
    # def auto_speech_set(self, value):
    #     """
    #     This method sets the auto_speech_flag and chunk_flag to the given value and False respectively, and prints the state of auto_speech_flag.
    #     The auto_speech_flag is used to control whether the speech input should be automatically processed.

    #     Args:
    #         value (bool): The value to set the auto_speech_flag to.
    #     """
    #     self.auto_speech_flag = value
    #     self.chunk_flag = False
    #     print(f"auto_speech_flag FLAG STATE: {self.auto_speech_flag}")

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
    def voice(self, flag):
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
            print(self.colors["OKCYAN"] + "🎙️ You can press shift+alt to interrupt speech generation. 🎙️" + self.colors["OKCYAN"])
           
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
            #TODO REFACTOR FOR NEW SYSTEM
        """
        # Search for the name after 'forward slash voice swap'
        # print(f"Agent voice swapped to {self.voice_name}")
        # print(self.colors['GREEN'] + f"<<< USER >>> " + self.colors['OKGREEN'])
        # return
        
    # -------------------------------------------------------------------------------------------------   
    def update_speech_flags(self, value):
        self.listen_flag = value
        self.speech_recognizer_instance.auto_speech_flag = value
        print(f"Speech recognition {'activated' if value else 'deactivated'}")
        
    # -------------------------------------------------------------------------------------------------   
    def listen(self):
        """ a method for changing the listen flag 
            args: flag
            return: none
        """
        # if not self.listen_flag:
        #     self.listen_flag = True
        #     print(self.colors["OKBLUE"] + "- speech to text activated -" + self.colors["RED"])
        #     print(self.colors["OKCYAN"] + "🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️" + self.colors["OKCYAN"])
        # else:
        #     print(self.colors["OKBLUE"] + "- speech to text deactivated -" + self.colors["RED"])

        # return
        self.update_speech_flags(not self.listen_flag)
        return True

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
    def wake_commands(self, flag):
        """ a method for auto_command flag 
            args: flag
            return: none
        """
        self.speech_recognizer_instance.use_wake_commands = flag
        print(f"use_wake_commands FLAG STATE: {self.speech_recognizer_instance.use_wake_commands}")
        return
    
    # # -------------------------------------------------------------------------------------------------    
    # def interrupt_speech(self):
    #     self.speech_interrupted = True
    #     if hasattr(self, 'tts_processor_instance'):
    #         # Stop any currently playing audio 
    #         sd.stop()
    #         # cut off speech generation as well
    #         self.tts_processor_instance.interrupt_generation()
    #     self.listen_flag = False  # Reset the listen flag
    
    # # ------------------------------------------------------------------------------------------------- 
    # def chatbot_main1(self):
    #     self.latex_render_instance = None
    #     self.tts_processor_instance = None

    #     keyboard.add_hotkey('ctrl+shift', self.speech_recognizer_instance.auto_speech_set, args=(True, self.listen_flag))
    #     keyboard.add_hotkey('ctrl+alt', self.speech_recognizer_instance.chunk_speech, args=(True,))
    #     keyboard.add_hotkey('shift+alt', self.interrupt_speech)
    #     keyboard.add_hotkey('tab+ctrl', self.speech_recognizer_instance.toggle_wake_commands)

    #     while True:
    #         user_input_prompt = ""
    #         speech_done = False
    #         cmd_run_flag = False
            
    #         if self.listen_flag:
    #             keyboard.wait('ctrl+shift')
                
    #             self.speech_recognizer_instance.auto_speech_flag = True
    #             while self.speech_recognizer_instance.auto_speech_flag:
    #                 try:
    #                     if self.listen_flag:
    #                         if self.speech_recognizer_instance.use_wake_commands:
    #                             user_input_prompt = self.speech_recognizer_instance.wake_words(audio)
    #                         else:
    #                             audio = self.speech_recognizer_instance.get_audio()
    #                             user_input_prompt = self.speech_recognizer_instance.recognize_speech(audio)
                                
    #                         if user_input_prompt:
    #                             self.speech_recognizer_instance.chunk_flag = False
    #                             self.speech_recognizer_instance.auto_speech_flag = False
                                
    #                             user_input_prompt = self.voice_command_select_filter(user_input_prompt)
    #                             cmd_run_flag = self.command_select(user_input_prompt)
                                
    #                             if self.listen_flag and not cmd_run_flag:
    #                                 response = self.send_prompt(user_input_prompt)
    #                                 response_processed = False
    #                                 if self.listen_flag is False and self.leap_flag is not None and isinstance(self.leap_flag, bool):
    #                                     if not self.leap_flag and not response_processed:
    #                                         self.tts_processor_instance.process_tts_responses(response, self.voice_name)
    #                                         response_processed = True
    #                                         if self.speech_interrupted:
    #                                             print("Speech was interrupted. Ready for next input.")
    #                                             self.speech_interrupted = False
    #                             break
                                
    #                 except sr.UnknownValueError:
    #                     print(self.colors["OKCYAN"] + "Google Speech Recognition could not understand audio" + self.colors["OKCYAN"])
                    
    #                 except sr.RequestError as e:
    #                     print(self.colors["OKCYAN"] + "Could not request results from Google Speech Recognition service; {0}".format(e) + self.colors["OKCYAN"])
                        
    #         if not self.listen_flag:
    #             user_input_prompt = input(self.colors["GREEN"] + f"<<< 🧠 USER 🧠 >>> " + self.colors["END"])
    #             speech_done = True
            
    #         user_input_prompt = self.voice_command_select_filter(user_input_prompt)
    #         cmd_run_flag = self.command_select(user_input_prompt)
            
    #         if self.llava_flag:
    #             self.screen_shot_flag = self.screen_shot_collector_instance.get_screenshot()
                
    #         if self.splice_flag:
    #             self.data_set_video_process_instance.generate_image_data()
            
    #         if not cmd_run_flag and speech_done:
    #             print(self.colors["YELLOW"] + f"{user_input_prompt}" + self.colors["OKCYAN"])
                
    #             response = self.send_prompt(user_input_prompt)

    #             if self.latex_flag:
    #                 latex_render_instance = latex_render_class()
    #                 latex_render_instance.add_latex_code(response, self.user_input_model_select)

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

        keyboard.add_hotkey('ctrl+shift', self.speech_recognizer_instance.auto_speech_set, args=(True, self.listen_flag))
        keyboard.add_hotkey('ctrl+alt', self.speech_recognizer_instance.chunk_speech, args=(True,))
        keyboard.add_hotkey('shift+alt', self.interrupt_speech)
        keyboard.add_hotkey('tab+ctrl', self.speech_recognizer_instance.toggle_wake_commands)

        while True:
            user_input_prompt = ""
            speech_done = False
            cmd_run_flag = False
            
            # check for speech recognition
            # if self.listen_flag or self.speech_recognizer_instance.auto_speech_flag:
            if self.listen_flag:
                # user input speech request from keybinds
                # Wait for the key press to start speech recognition
                keyboard.wait('ctrl+shift')
                
                # Start speech recognition
                self.speech_recognizer_instance.auto_speech_flag = True
                while self.speech_recognizer_instance.auto_speech_flag:
                    try:
                        # Record audio from microphone
                        if self.listen_flag:
                            # Recognize speech to text from audio
                            if self.speech_recognizer_instance.use_wake_commands:
                                # using wake commands
                                user_input_prompt = self.speech_recognizer_instance.wake_words(audio)
                            else:
                                audio = self.speech_recognizer_instance.get_audio()
                                # using push to talk
                                user_input_prompt = self.speech_recognizer_instance.recognize_speech(audio)
                                
                            # print recognized speech
                            if user_input_prompt:
                                self.speech_recognizer_instance.chunk_flag = False
                                self.speech_recognizer_instance.auto_speech_flag = False
                                
                                # Filter voice commands and execute them if necessary
                                user_input_prompt = self.voice_command_select_filter(user_input_prompt)
                                cmd_run_flag = self.command_select(user_input_prompt)
                                
                                # Check if the listen flag is still on before sending the prompt to the model
                                if self.listen_flag and not cmd_run_flag:
                                    # Send the recognized speech to the model
                                    response = self.send_prompt(user_input_prompt)
                                    # Process the response with the text-to-speech processor
                                    response_processed = False
                                    if self.listen_flag is False and self.leap_flag is not None and isinstance(self.leap_flag, bool):
                                        if not self.leap_flag and not response_processed:
                                            self.tts_processor_instance.process_tts_responses(response, self.voice_name)
                                            response_processed = True
                                            if self.speech_interrupted:
                                                print("Speech was interrupted. Ready for next input.")
                                                self.speech_interrupted = False
                                break  # Exit the loop after recognizing speech
                                
                    # google speech recognition error exception: inaudible sample
                    except sr.UnknownValueError:
                        print(self.colors["OKCYAN"] + "Google Speech Recognition could not understand audio" + self.colors["OKCYAN"])
                    
                    # google speech recognition error exception: no connection
                    except sr.RequestError as e:
                        print(self.colors["OKCYAN"] + "Could not request results from Google Speech Recognition service; {0}".format(e) + self.colors["OKCYAN"])
                        
            # if speech recognition is off, request text input from user
            if not self.listen_flag:
                user_input_prompt = input(self.colors["GREEN"] + f"<<< 🧠 USER 🧠 >>> " + self.colors["END"])
                speech_done = True
            
            # filter voice cmds -> parse and execute user input commands
            user_input_prompt = self.voice_command_select_filter(user_input_prompt)
            cmd_run_flag = self.command_select(user_input_prompt)
            
            # get screenshot
            if self.llava_flag:
                self.screen_shot_flag = self.screen_shot_collector_instance.get_screenshot()
                
            # splice videos
            if self.splice_flag:
                self.data_set_video_process_instance.generate_image_data()
            
            # if conditional, send prompt to assistant
            if not cmd_run_flag and speech_done:
                print(self.colors["YELLOW"] + f"{user_input_prompt}" + self.colors["OKCYAN"])
                
                # Send the prompt to the assistant
                response = self.send_prompt(user_input_prompt)

                # Process the response with the text-to-speech processor
                # response_processed = False
                # if self.listen_flag is False and self.leap_flag is not None and isinstance(self.leap_flag, bool):
                #     if not self.leap_flag and not response_processed:
                #         self.tts_processor_instance.process_tts_responses(response, self.voice_name)
                #         response_processed = True
                #         if self.speech_interrupted:
                #             print("Speech was interrupted. Ready for next input.")
                #             self.speech_interrupted = False

                # Check for latex and add to queue
                if self.latex_flag:
                    # Create a new instance
                    latex_render_instance = latex_render_class()
                    latex_render_instance.add_latex_code(response, self.user_input_model_select)
                    
import logging
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import ollama
import json
import asyncio

from ollama_chatbot_base import ollama_chatbot_base

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    type: str

class ModelRequest(BaseModel):
    model: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

chatbot = ollama_chatbot_base()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected"
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data}")
                
                json_data = json.loads(data)
                message_type = json_data.get('type')
                content = json_data.get('message')

                if not message_type or not content:
                    raise ValueError("Invalid message format")

                if message_type == 'chat':
                    if not chatbot.user_input_model_select:
                        await websocket.send_json({
                            "type": "error",
                            "message": "No model selected. Please select a model first."
                        })
                        continue

                    chatbot.chat_history.append({"role": "user", "content": content})
                    
                    try:
                        response = ollama.chat(
                            model=chatbot.user_input_model_select,
                            messages=chatbot.chat_history,
                            stream=True
                        )
                        
                        full_response = ''
                        for chunk in response:
                            if 'message' in chunk and 'content' in chunk['message']:
                                content_chunk = chunk['message']['content']
                                full_response += content_chunk
                                
                                await websocket.send_json({
                                    "type": "chat_response",
                                    "response": content_chunk
                                })
                                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the frontend
                        
                        chatbot.chat_history.append({"role": "assistant", "content": full_response})
                        
                        await websocket.send_json({
                            "type": "chat_response_end",
                            "response": full_response
                        })
                        
                        if not chatbot.leap_flag:
                            await chatbot.tts_processor_instance.process_tts_responses(full_response, chatbot.voice_name)
                        
                    except Exception as e:
                        logger.error(f"Error in chat response: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Chat error: {str(e)}"
                        })
                        
                elif message_type == 'command':
                    try:
                        result = chatbot.command_select(content)
                        await  websocket.send_json({
                            "type": "command_result",
                            "response": str(result)
                        })
                    except Exception as e:
                        logger.error(f"Error executing command: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Command error: {str(e)}"
                        })
                else:
                    raise ValueError(f"Unknown message type: {message_type}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {data}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid JSON: {str(e)}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {str(e)}")
        manager.disconnect(websocket)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if request.message.startswith('/'):
            result = chatbot.command_select(request.message)
            return ChatResponse(response=result, type="command_result")
        else:
            response = await chatbot.send_prompt(request.message)
            return ChatResponse(response=response, type="chat_response")
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_model")
async def set_model(request: ModelRequest):
    try:
        chatbot.set_model(request.model)
        return {"message": f"Model set to {request.model}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current_model")
async def get_current_model():
    return {"model": chatbot.user_input_model_select}

@app.get("/available_models")
async def get_available_models():
    try:
        models = await chatbot.ollama_command_instance.ollama_list()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/command_library")
async def get_command_library():
    if hasattr(chatbot, 'command_library'):
        return {"commands": list(chatbot.command_library.keys())}
    else:
        return {"commands": []}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.websocket("/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_audio_data = chatbot.get_user_audio_data()
            llm_audio_data = chatbot.get_llm_audio_data()
            await websocket.send_json({
                "user_audio_data": user_audio_data.tolist(),
                "llm_audio_data": llm_audio_data.tolist()
            })
            await asyncio.sleep(0.1)  # Send audio data every 100ms
    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")

@app.post("/hotkeys/{action}")
async def manage_hotkeys(action: str):
    if action == "setup":
        chatbot.setup_hotkeys()
        return {"message": "Hotkeys set up"}
    elif action == "remove":
        chatbot.remove_hotkeys()
        return {"message": "Hotkeys removed"}
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

@app.get("/speech_recognition_status")
async def get_speech_recognition_status():
    return {"active": chatbot.speech_recognition_active}

@app.get("/ollama/loaded_models")
async def get_loaded_models():
    return await chatbot.ollama_command_instance.ollama_show_loaded_models()

@app.get("/ollama/template")
async def get_template():
    return await chatbot.ollama_command_instance.ollama_show_template()

@app.get("/ollama/license")
async def get_license():
    return await chatbot.ollama_command_instance.ollama_show_license()

@app.get("/ollama/modelfile")
async def get_modelfile():
    return await chatbot.ollama_command_instance.ollama_show_modelfile()

@app.get("/ollama/list")
async def get_ollama_list():
    return await chatbot.ollama_command_instance.ollama_list()

@app.post("/ollama/create")
async def create_ollama_model():
    return await chatbot.ollama_command_instance.ollama_create()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2020)