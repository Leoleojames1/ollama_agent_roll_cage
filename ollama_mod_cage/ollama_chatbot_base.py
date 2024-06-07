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

from Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands
from Public_Chatbot_Base_Wand.speech_to_speech import tts_processor_class
from Public_Chatbot_Base_Wand.directory_manager import directory_manager_class
from Public_Chatbot_Base_Wand.latex_render import latex_render_class
from Public_Chatbot_Base_Wand.data_set_manipulator import data_set_constructor
from Public_Chatbot_Base_Wand.write_modelfile import model_write_class
from Public_Chatbot_Base_Wand.chat_history import json_chat_history
from Public_Chatbot_Base_Wand.read_write_symbol_collector import read_write_symbol_collector
from Public_Chatbot_Base_Wand.data_set_manipulator import screen_shot_collector
from Public_Chatbot_Base_Wand.create_convert_model import create_convert_manager

# TODO setup sebdg emotional classifyer keras 
# from tensorflow.keras.models import load_model
# sentiment_model = load_model('D:\\CodingGit_StorageHDD\\model_git\\emotions_classifier\\emotions_classifier.keras')

# -------------------------------------------------------------------------------------------------
class ollama_chatbot_base:
    """ A class for accessing the ollama local serve api via python, and creating new custom agents.
    The ollama_chatbot_class is also used for accessing Speech to Text transcription/Text to Speech Generation methods via a speedy
    low level, command line interface and the Tortoise TTS model.
    """

    # -------------------------------------------------------------------------------------------------
    def __init__(self, wizard_name):
        """ a method for initializing the ollama_chatbot_base class 
            Args: user_unput_model_select
            Returns: none
        """
        # get user input model selection
        self.get_model()
        self.user_input_model_select = self.user_input_model_select
        self.wizard_name = wizard_name

        # initialize chat
        self.chat_history = []
        self.llava_history = []

        # Default Agent Voice Reference
        self.voice_name = "C3PO"

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

        # build conversation save path
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
        # get data
        self.screen_shot_collector_instance = screen_shot_collector(self.developer_tools_dict)
        self.json_chat_history_instance = json_chat_history(self.developer_tools_dict)
        self.data_set_video_process_instance = data_set_constructor(self.developer_tools_dict)
        # generate
        self.model_write_class_instance = model_write_class(self.colors, self.developer_tools_dict)
        self.create_convert_manager_instance = create_convert_manager(self.colors, self.developer_tools_dict)

    # -------------------------------------------------------------------------------------------------  
    def get_model(self):
        """ a method for collecting the model name from the user input
        """
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        self.user_input_model_select = input(HEADER + "<<< PROVIDE AGENT NAME >>> " + OKBLUE)
    
    # -------------------------------------------------------------------------------------------------
    def swap(self):
        """ a method to call when swapping models
        """
        self.chat_history = []
        self.user_input_model_select = input(self.colors['HEADER']+ "<<< PROVIDE AGENT NAME TO SWAP >>> " + self.colors['OKBLUE'])
        print(f"Model changed to {self.user_input_model_select}")
        return
    
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
    def send_prompt(self, user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        #TODO ADD IF MEM OFF CLEAR HISTORY
        self.chat_history = []
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

        # Parse for Token Specific Arg Commands
        # Parse for the name after 'forward slash voice swap'
        match = re.search(r"(activate voice swap|/voice swap) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.voice_name = match.group(2)
            self.voice_name = self.tts_processor_instance.file_name_conversation_history_filter(self.voice_name)

        # Parse for the name after 'forward slash movie'
        match = re.search(r"(activate movie|/movie) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.movie_name = match.group(2)
            self.movie_name = self.tts_processor_instance.file_name_conversation_history_filter(self.movie_name)
        else:
            self.movie_name = None

        # Parse for the name after 'activate save'
        match = re.search(r"(activate save as|/save as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.save_name = match.group(2)
            self.save_name = self.tts_processor_instance.file_name_conversation_history_filter(self.save_name)
            print(f"save_name string: {self.save_name}")
        else:
            self.save_name = None

        # Parse for the name after 'activate load'
        match = re.search(r"(activate load as|/load as) ([^/.]*)", user_input_prompt, flags=re.IGNORECASE)
        if match:
            self.load_name = match.group(2)
            self.load_name = self.tts_processor_instance.file_name_conversation_history_filter(self.load_name)
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
        """ a method for selecting the command to execute
            Args: command_str
            Returns: command_library[command_str]
        """
        command_library = {
            "/swap": lambda: self.swap(),
            "/voice swap": lambda: self.voice_swap(),
            "/save as": lambda: self.json_chat_history_instance.save_to_json(),
            "/load as": lambda: self.json_chat_history_instance.load_from_json(),
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
            "/ollama create": lambda: self.ollama_command_instance.ollama_create(),
            "/ollama show": lambda: self.ollama_command_instance.ollama_show_modelfile(),
            "/ollama template": lambda: self.ollama_command_instance.ollama_show_template(),
            "/ollama license": lambda: self.ollama_command_instance.ollama_show_license(),
            "/ollama list": lambda: self.ollama_command_instance.ollama_list(),
            "/splice video": lambda: self.data_set_video_process_instance.generate_image_data(),
            "/developer new" : lambda: self.read_write_symbol_collector_instance.developer_tools_generate()
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
        speech_str = sr.Recognizer().recognize_google(audio)
        print(f">>{speech_str}<<")
        return speech_str
    
    # -------------------------------------------------------------------------------------------------
    def chatbot_main(self):
        """ a method for managing the current chatbot instance loop 
            args: None
            returns: None
        """
        # wait to load tts & latex until needed
        self.latex_render_instance = None
        self.tts_processor_instance = None

        print(self.colors["OKCYAN"] + "Press space bar to record audio:" + self.colors["OKCYAN"])
        print(self.colors["GREEN"] + f"<<< USER >>> " + self.colors["END"])

        keyboard.add_hotkey('ctrl+a+d', self.auto_speech_set, args=(True,))
        keyboard.add_hotkey('ctrl+s+w', self.chunk_speech, args=(True,))

        while True:
            user_input_prompt = ""
            speech_done = False
            cmd_run_flag = False

            if self.listen_flag | self.auto_speech_flag is True:
                self.tts_processor_instance = self.instance_tts_processor()
                while self.auto_speech_flag is True:  # user holds down the space bar
                    try:
                        # Record audio from microphone
                        audio = self.get_audio()
                        if self.listen_flag is True:
                            # Recognize speech to text from audio
                            user_input_prompt = self.recognize_speech(audio)
                            print(f">>SPEECH RECOGNIZED<< >> {user_input_prompt} <<")
                            speech_done = True
                            self.chunk_flag = False
                            print(f"CHUNK FLAG STATE: {self.chunk_flag}")
                            self.auto_speech_flag = False
                    except sr.UnknownValueError:
                        print(self.colors["OKCYAN"] + "Google Speech Recognition could not understand audio" + self.colors["OKCYAN"])
                    except sr.RequestError as e:
                        print(self.colors["OKCYAN"] + "Could not request results from Google Speech Recognition service; {0}".format(e) + self.colors["OKCYAN"])
            elif self.listen_flag is False:
                print(self.colors["OKCYAN"] + "Please type your selected prompt:" + self.colors["OKCYAN"])
                user_input_prompt = input(self.colors["GREEN"] + f"<<< USER >>> " + self.colors["END"])
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
                print(self.colors["RED"] + f"<<< {self.user_input_model_select} >>> " + self.colors["RED"] + f"{response}" + self.colors["RED"])
                # Check for latex and add to queue
                if self.latex_flag:
                    # Create a new instance
                    latex_render_instance = latex_render_class()
                    latex_render_instance.add_latex_code(response, self.user_input_model_select)
                # Preprocess for text to speech, add flag for if text to speech enable handle canche otherwise do /leap or smt
                # Clear speech cache and split the response into sentences for next TTS cache
                if self.leap_flag is not None and isinstance(self.leap_flag, bool):
                    if self.leap_flag != True:
                        self.tts_processor_instance.process_tts_responses(response, self.voice_name)
                elif self.leap_flag is None:
                    pass
                # Start the mainloop in the main thread
                print(self.colors["GREEN"] + f"<<< USER >>> " + self.colors["END"])

    # -------------------------------------------------------------------------------------------------   
    def chunk_speech(self, value):
        # time.sleep(1)
        self.chunk_flag = value
        print(f"chunk_flag FLAG STATE: {self.chunk_flag}")

    # -------------------------------------------------------------------------------------------------   
    def auto_speech_set(self, value):
        self.auto_speech_flag = value
        self.chunk_flag = False
        print(f"auto_speech_flag FLAG STATE: {self.auto_speech_flag}")

    # -------------------------------------------------------------------------------------------------
    def instance_tts_processor(self):
        if not hasattr(self, 'tts_processor_instance') or self.tts_processor_instance is None:
            self.tts_processor_instance = tts_processor_class(self.colors, self.developer_tools_dict)
        return self.tts_processor_instance
    
    # -------------------------------------------------------------------------------------------------   
    def leap(self, flag):
        """ a method for changing the leap flag 
            args: flag
            returns: none
        """
        self.leap_flag = flag
        print(f"leap_flag FLAG STATE: {self.leap_flag}")
        return
    
    # -------------------------------------------------------------------------------------------------   
    def speech(self, flag1, flag2):
        """ a method for changing the speech to speech flags 
            args: flag1, flag2
            returns: none
        """
        if flag2 == True:
            self.tts_processor_instance = self.instance_tts_processor()
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
        if flag == True:
            self.tts_processor_instance = self.instance_tts_processor()
        self.listen_flag = flag
        print(f"listen_flag FLAG STATE: {self.listen_flag}")
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