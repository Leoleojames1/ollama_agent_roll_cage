""" driver_script.py

    driver_script.py is the driver for ollama_agent_roll_cage, is a command line interface for STT, 
    & TTS commands with local LLMS. It is an easy to install add on for the ollama application.
    
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

import keyboard
import speech_recognition as sr

from ollama_chatbot_class import ollama_chatbot_class
from Public_Chatbot_Base_Wand.flags import flag_manager
from Public_Chatbot_Base_Wand.ollama_add_on_library import ollama_commands
from Public_Chatbot_Base_Wand.speech_to_speech import tts_processor_class
from Public_Chatbot_Base_Wand.directory_manager import directory_manager_class
from Public_Chatbot_Base_Wand.latex_render import latex_render_class
from Public_Chatbot_Base_Wand.data_set_manipulator import data_set_constructor
from Public_Chatbot_Base_Wand.write_modelfile import model_write_class
from Public_Chatbot_Base_Wand.chat_history import json_chat_history
from Public_Chatbot_Base_Wand.read_write_symbol_collector import read_write_symbol_collector

# -------------------------------------------------------------------------------------------------
class driver_setup:
    """ a class for setting up the class tool instances and mod tool instances for the defined chatbot instances
    """
    # -------------------------------------------------------------------------------------------------
    def __init__(self):
        test ="test"
    # -------------------------------------------------------------------------------------------------
    def instance_tts_processor(self):
        if not hasattr(self, 'tts_processor_instance') or self.tts_processor_instance is None:
            self.tts_processor_instance = tts_processor_class()
        return self.tts_processor_instance
    
    # -------------------------------------------------------------------------------------------------
    def instance_latex_render(self):
        if not hasattr(self, 'latex_render_instance') or self.latex_render_instance is None:
            self.latex_render_instance = latex_render_class()
        return self.latex_render_instance
    
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    """ 
    The main loop for the ollama_chatbot_class, utilizing a state machine for user command injection during command line prompting,
    all commands start with /, and are named logically.
    """
    driver_setup_instance = None
    flag_manager_instance = None
    ollama_command_instance = None
    chat_history_instance = None
    model_write_class_instance = None
    read_write_symbol_collector_instance = None
    latex_render_instance = None
    data_set_video_process_instance = None
    tts_processor_instance = None

    instances = {
        driver_setup_instance : driver_setup(),
        flag_manager_instance : flag_manager(),
        ollama_command_instance : ollama_commands(),
        chat_history_instance : json_chat_history(),
        model_write_class_instance : model_write_class(),
        read_write_symbol_collector_instance : read_write_symbol_collector()
    }

    colors = ollama_command_instance.get_colors()
    screen_shot_flag = False

    # new instance class
    ollama_chatbot_class_instance = ollama_chatbot_class(driver_setup_instance)
    instances[data_set_video_process_instance] = data_set_constructor()

    # select agent name
    ollama_chatbot_class_instance.user_input_model_select = input(colors["HEADER"] + "<<< PROVIDE AGENT NAME >>> " + colors["OKBLUE"])


    print(colors["OKCYAN"] + "Press space bar to record audio:" + colors["OKCYAN"])
    print(colors["GREEN"] + f"<<< USER >>> " + colors["END"])
    # keyboard.add_hotkey('ctrl+a+d', print, args=('triggered', 'begin speech'))
    # keyboard.add_hotkey('ctrl+a+d', print, args=('triggered', 'begin speech'))

    # def chunk_speach(value):
    #     ollama_chatbot_class.chunk_flag = value
    #     print(f"CHUNK FLAG STATE: {ollama_chatbot_class.listen_flag}")

    keyboard.add_hotkey('ctrl+a+d', ollama_chatbot_class_instance.auto_speech_set, args=(True,))
    keyboard.add_hotkey('ctrl+s+w', ollama_chatbot_class_instance.chunk_speech, args=(True,))

    while True:
        user_input_prompt = ""
        speech_done = False
        cmd_run_flag = False
        
        # print(f"WHILE LOOP WAY TOP LISTEN: {ollama_chatbot_class.listen_flag}")
        # print(f"WHILE LOOP WAY TOP AUTO: {ollama_chatbot_class.auto_speech_flag}")
        # print(f"WHILE LOOP WAY TOP CHUNK: {ollama_chatbot_class.chunk_flag}")

        if ollama_chatbot_class_instance.listen_flag | ollama_chatbot_class_instance.auto_speech_flag is True:
            instances[tts_processor_instance] = driver_setup_instance.instance_tts_processor()
            # print(f"ENTER IF LISTEN TRUE LISTEN: {ollama_chatbot_class.listen_flag}") 
            # print(f"ENTER IF LISTEN TRUE AUTO: {ollama_chatbot_class.auto_speech_flag}") 
            # print(f"ENTER IF LISTEN TRUE CHUNK: {ollama_chatbot_class.chunk_flag}")
            while ollama_chatbot_class_instance.auto_speech_flag is True:  # user holds down the space bar
                try:
                    # Record audio from microphone
                    audio = instances[tts_processor_instance].get_audio(ollama_chatbot_class_instance)

                    if ollama_chatbot_class_instance.listen_flag is True:
                        # Recognize speech to text from audio
                        user_input_prompt = tts_processor_instance.recognize_speech(audio)
                        print(f">>SPEECH RECOGNIZED<< >> {user_input_prompt} <<")
                        speech_done = True
                        ollama_chatbot_class_instance.chunk_flag = False
                        print(f"CHUNK FLAG STATE: {ollama_chatbot_class_instance.chunk_flag}")
                        ollama_chatbot_class_instance.auto_speech_flag = False

                except sr.UnknownValueError:
                    print(colors["OKCYAN"] + "Google Speech Recognition could not understand audio" + colors["OKCYAN"])
                except sr.RequestError as e:
                    print(colors["OKCYAN"] + "Could not request results from Google Speech Recognition service; {0}".format(e) + colors["OKCYAN"])
        elif ollama_chatbot_class_instance.listen_flag is False:
            print(colors["OKCYAN"] + "Please type your selected prompt:" + colors["OKCYAN"])
            user_input_prompt = input(colors["GREEN"] + f"<<< USER >>> " + colors["END"])
            speech_done = True

        # Use re.sub to replace "forward slash cmd" with "/cmd"
        # print(f"MID ELIF LISTEN: {ollama_chatbot_class.listen_flag}")
        # print(f"MID ELIF AUTO: {ollama_chatbot_class.auto_speech_flag}")
        # print(f"MID ELIF CHUNK: {ollama_chatbot_class.chunk_flag}")

        user_input_prompt = ollama_chatbot_class_instance.voice_command_select_filter(user_input_prompt)
        cmd_run_flag = ollama_chatbot_class_instance.command_select(user_input_prompt, flag_manager_instance, ollama_chatbot_class_instance)
        
        # get screenshot
        if ollama_chatbot_class_instance.llava_flag is True:
            screen_shot_flag = ollama_chatbot_class_instance.get_screenshot()
        # splice videos
        if ollama_chatbot_class_instance.splice_flag == True:
            data_set_video_process_instance.generate_image_data()

        if cmd_run_flag == False and speech_done == True:
            print(colors["YELLOW"] + f"{user_input_prompt}" + colors["OKCYAN"])

            # Send the prompt to the assistant
            if screen_shot_flag is True:
                response = ollama_chatbot_class_instance.send_prompt(user_input_prompt)
                screen_shot_flag = False
            else:
                response = ollama_chatbot_class_instance.send_prompt(user_input_prompt)
            
            print(colors["RED"] + f"<<< {ollama_chatbot_class_instance.user_input_model_select} >>> " + colors["RED"] + f"{response}" + colors["RED"])

            # Check for latex and add to queue
            if ollama_chatbot_class_instance.latex_flag:
                # Create a new instance
                latex_render_instance = latex_render_class()
                latex_render_instance.add_latex_code(response, ollama_chatbot_class_instance.user_input_model_select)

            # Preprocess for text to speech, add flag for if text to speech enable handle canche otherwise do /leap or smt
            # Clear speech cache and split the response into sentences for next TTS cache
            if ollama_chatbot_class_instance.leap_flag is not None and isinstance(ollama_chatbot_class_instance.leap_flag, bool):
                if ollama_chatbot_class_instance.leap_flag != True:
                    tts_processor_instance.process_tts_responses(response, ollama_chatbot_class_instance.voice_name)
            elif ollama_chatbot_class_instance.leap_flag is None:
                pass
            # Start the mainloop in the main thread
            print(colors["GREEN"] + f"<<< USER >>> " + colors["END"])