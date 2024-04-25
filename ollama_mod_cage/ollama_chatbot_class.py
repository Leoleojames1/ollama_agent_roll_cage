""" ollama_chatbot_class.py

    ollama_agent_roll_cage, is a command line interface for STT, TTS commands with local LLMS.
    It is an easy to install add on for the ollama application.
    
        This software was designed by Leo Borcherding with the intent of creating an easy to use
    ai interface for anyone, through Speech to Text and Text to Speech.
        
        With ollama_agent_roll_cage we can provide hands free access to LLM data. 
    This has a host of applications and Im excited to bring this software to users 
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
import requests
import json
import torch
import sounddevice as sd
from TTS.api import TTS
import re
import speech_recognition as sr
import keyboard

class ollama_chatbot_class:
    """ A class for accessing the ollama local serve api via python, and creating new custom agents.
    The ollama_chatbot_class is also used for accessing Speech to Text transcription/Text to Speech Generation methods via a speedy
    low level, command line interface and the Tortoise TTS model.
    """
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.headers = {'Content-Type': 'application/json'}
        self.chat_history = []

        current_dir = os.getcwd()
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_wav_path = os.path.join(parent_dir, "AgentFiles\\Ignored_TTS\\pipeline\\active_group\\clone_speech.wav")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def agent_text_to_voice(self, response):
        tts_response = self.tts.tts(text=response, speaker_wav=(f"{self.tts_wav_path}"), language="en")
        return tts_response
    
    def get_audio(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
        return audio

    def _generate_tts_response(self, response):
        return self.tts.tts(text=response, speaker_wav=self.tts_wav_path, language="en")
    
    def send_prompt(self, user_input_prompt, user_input_model_select):
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

    def split_into_sentences(self, text: str) -> list[str]:
        # Add spaces around punctuation marks for consistent splitting
        text = " " + text + " "
        text = text.replace("\n", " ")

        # Handle common abbreviations and special cases
        text = re.sub(r"(Mr|Mrs|Ms|Dr|i\.e)\.", r"\1<prd>", text)
        text = re.sub(r"\.\.\.", r"<prd><prd><prd>", text)

        # Split on period, question mark, or exclamation mark followed by optional spaces
        sentences = re.split(r"[.!?]\s*", text)

        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences
    
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
    
    TODO REGEX COMMANDS: ---------------------------------------------
    ollama_agent_roll_cage 0.2 UPDATE:

    /create {custom system prompt} (current model)
    /speech {modify the STT settings} {Push To Talk} {Long Listen}
    /text {modify the TTS settings}        
    /search {USER GOOGLE SEARCH REQUEST}
    elif user_input_prompt.lower() CONTAINS {/search}, use REGEX "/search {USER SEARCH REQUEST}":
        then make google search api request from local machine and provide that data to the model to digest
        collect /search detection and {USER SEARCH REQUEST} data via regex match groups
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
        user_input_prompt = ""
        speech_done = False
        while keyboard.is_pressed('space'):  # user holds down the space bar
            audio = ollama_chatbot_class.get_audio()
            try:
                user_input_prompt = sr.Recognizer().recognize_google(audio)
                speech_done = True
                print(GREEN + f"<<< USER >>> ")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

        # If speech recognition was successful, proceed with the rest of the loop
        if speech_done:
            print(YELLOW + f"{user_input_prompt}" + OKCYAN)
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
                user_input_model_select = input(HEADER + "<<< PROVIDE NEW AGENT NAME >>> " + OKBLUE)
                print(f"Model changed to {user_input_model_select}")

            # Quit back to root directory in cmd
            elif user_input_prompt.lower() == "/quit":
                break

            # if not command, then prompt model
            else:
                # Send the prompt to the assistant
                response = ollama_chatbot_class.send_prompt(user_input_prompt, user_input_model_select)
                print(RED + f"<<< {user_input_model_select} >>> " + END + f"{response}" + RED)

                # Split the response into sentences for TTS
                tts_response_sentences = ollama_chatbot_class.split_into_sentences(response)

                # Generate audio for each sentence in TTS
                for sentence in tts_response_sentences:
                    tts_audio = ollama_chatbot_class.agent_text_to_voice(sentence)
                    # Play the audio
                    sd.play(tts_audio, samplerate=22050)
                    sd.wait()