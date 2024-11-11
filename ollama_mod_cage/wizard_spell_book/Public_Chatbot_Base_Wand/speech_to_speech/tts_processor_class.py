""" tts_processor.py
        
        A class for processing the response sentences and audio generation for the 
        ollama_chat_bot_base through hotkeys, flag managers, and a smart speech
        rendering queue designed to optimize speech speed and speech quality.

created on: 4/20/2024
by @LeoBorcherding
"""

import sounddevice as sd
import soundfile as sf
import threading
import os
import torch
import re
import queue

# TODO import new eginhard fork
# eginhard fork of coqui provides constant updates and 
# advanced features for coqui post shutdown
# from Garden_chatbot_Base_Wand.coqui import TTS

# TODO import F5 TTS model, setup coqui bark, tortoise, piper & more

# TODO look into setting up whisper speech, with whisper for
# custom audio embedding token access

from TTS.api import TTS
import numpy as np
import shutil
import time
import keyboard

# -------------------------------------------------------------------------------------------------
class tts_processor_class:
    """ a class for managing the text to speech conversation between the user, ollama, & coqui-tts.
    """
    # # -------------------------------------------------------------------------------------------------
    # def __init__(self, colors, developer_tools_dict, voice_type, voice_name):
    #     """a method for initializing the class
    #     """ 
    #     self.voice_type = voice_type
    #     self.voice_name = voice_name
    #     self.colors = colors
    #     self.is_multi_speaker = None
        
    #     # get file paths from developer tools dictionary
    #     self.developer_tools_dict = developer_tools_dict
    #     self.current_dir = developer_tools_dict['current_dir']
    #     self.parent_dir = developer_tools_dict['parent_dir']
    #     self.speech_dir = developer_tools_dict['speech_dir']
    #     self.recognize_speech_dir = developer_tools_dict['recognize_speech_dir']
    #     self.generate_speech_dir = developer_tools_dict['generate_speech_dir']
    #     self.tts_voice_ref_wav_pack_path = developer_tools_dict['tts_voice_ref_wav_pack_path_dir']
    #     self.conversation_library = developer_tools_dict['conversation_library_dir']
    #     self.voice_name_reference_speech_path = None  # Initialize the attribute
    #     self.audio_data = np.array([])  # Initialize audio data buffer
        
    #     if torch.cuda.is_available():
    #         self.device = "cuda"
    #     else:
    #         self.device = "cpu"
    #         print("CUDA-compatible GPU is not available. Using CPU instead. If you believe this should not be the case, reinstall torch-audio with the correct version.")

    #     #TODO XTTS fine-tune selector and path manager
    #     # create instances for each voice model, tts processor should re instance if voice is changed, and if name is not in fine tune folder, then search for voice in default voice reference
        
    #     #======================
    #     #TODO for each xtts finetune print each name and ask user to select which to instance, 1, 2, 3, 4, ...

    #     # self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    #     # self.tts = \
    #     #         TTS(model_path=f"{self.parent_dir}/AgentFiles/Ignored_TTS/XTTS-v2_C3PO/", 
    #     #                     config_path=f"{self.parent_dir}/AgentFiles/Ignored_TTS/XTTS-v2_C3PO/config.json", progress_bar=False, gpu=True).to(self.device)

    #     #TODO implement default voice on startup with voice selector on /voice swap
    #     # self.tts = \
    #     #         TTS(model_path=f"{self.parent_dir}/AgentFiles/Ignored_TTS/XTTS-v2_CarliG/", 
    #     #                     config_path=f"{self.parent_dir}/AgentFiles/Ignored_TTS/XTTS-v2_CarliG/config.json", progress_bar=False, gpu=True).to(self.device)

    #     # ask for voice selection from user? or have default
    #     # print("initializing tts")
    #     # self.initialize_tts_model()

    #     fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
    #     fine_tuned_model_path = os.path.join(fine_tuned_dir, f"XTTS-v2_{self.voice_name}")
    #     reference_wav_path = os.path.join(fine_tuned_model_path, "reference.wav")
    #     print(f"{fine_tuned_model_path}")
        
    #     if os.path.exists(fine_tuned_model_path):
    #         # Use fine-tuned model (single-speaker)
    #         print("taking finetune path.")
    #         config_path = os.path.join(fine_tuned_model_path, "config.json")
    #         self.tts = TTS(model_path=fine_tuned_model_path, config_path=config_path, progress_bar=False, gpu=True).to(self.device)
    #         self.is_multi_speaker = False
    #         self.voice_name_reference_speech_path = reference_wav_path  # Not needed for fine-tuned model
    #     else:
    #         print("taking base xtts path.")
    #         # Use base XTTS model with voice reference (multi-speaker)
    #         self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
    #         self.is_multi_speaker = True
    #         self.voice_name_reference_speech_path = os.path.join(self.tts_voice_ref_wav_pack_path, self.voice_name, "clone_speech.wav")

    #     self.audio_queue = queue.Queue()
    
    # -------------------------------------------------------------------------------------------------
    def __init__(self, colors, developer_tools_dict, voice_type, voice_name):
        self.voice_type = voice_type
        self.voice_name = voice_name
        self.colors = colors
        self.is_multi_speaker = None
        self.speech_interrupted = False
        self.is_generating = False
        self.current_chunk_index = 0
        
        # Audio buffer for streaming
        self.audio_buffer = np.array([], dtype=np.float32)
        self.chunk_size = 1024
        self.sample_rate = 22050
        self.stream_buffer = queue.Queue()
        
        # Configure device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("CUDA not available. Using CPU for TTS.")
            
        # Initialize paths from developer tools
        self.setup_paths(developer_tools_dict)
        
        # Initialize TTS model
        self.initialize_tts_model()
        
        # Create audio stream
        self.audio_output_stream = None
        self.stream_active = False
        
    # -------------------------------------------------------------------------------------------------
    def setup_paths(self, developer_tools_dict):
        """Setup paths from developer tools dictionary"""
        self.developer_tools_dict = developer_tools_dict
        self.current_dir = developer_tools_dict['current_dir']
        self.parent_dir = developer_tools_dict['parent_dir']
        self.speech_dir = developer_tools_dict['speech_dir']
        self.recognize_speech_dir = developer_tools_dict['recognize_speech_dir']
        self.generate_speech_dir = developer_tools_dict['generate_speech_dir']
        self.tts_voice_ref_wav_pack_path = developer_tools_dict['tts_voice_ref_wav_pack_path_dir']
        
    # -------------------------------------------------------------------------------------------------
    def initialize_tts_model(self):
        """ a method to initialize the appropriate finetuned text to speech with coqui

            args: none
            returns: none
        """
        fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
        fine_tuned_model_path = os.path.join(fine_tuned_dir, f"XTTS-v2_{self.voice_name}")
        
        if os.path.exists(fine_tuned_model_path):
            # Use fine-tuned model
            config_path = os.path.join(fine_tuned_model_path, "config.json")
            self.tts = TTS(model_path=fine_tuned_model_path, 
                          config_path=config_path, 
                          progress_bar=False, 
                          gpu=True).to(self.device)
            self.is_multi_speaker = False
            self.voice_reference_path = os.path.join(fine_tuned_model_path, "reference.wav")
        else:
            # Use base model with reference
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            self.is_multi_speaker = True
            self.voice_reference_path = os.path.join(self.tts_voice_ref_wav_pack_path, 
                                                   self.voice_name, 
                                                   "clone_speech.wav")

    # # -------------------------------------------------------------------------------------------------
    # def process_tts_responses(self, response, voice_name):
    #     """A method for managing the response preprocessing methods.
    #         args: response, voice_name
    #         returns: none
    #     """
    #     # Clear VRAM cache
    #     torch.cuda.empty_cache()
    #     # Call Sentence Splitter
    #     tts_response_sentences = self.split_into_sentences(response)
        
    #     # Clear the directories
    #     self.clear_directory(self.recognize_speech_dir)
    #     self.clear_directory(self.generate_speech_dir)

    #     self.generate_play_audio_loop(tts_response_sentences)
    #     return
    
    # -------------------------------------------------------------------------------------------------
    def process_tts_responses(self, response, voice_name):  
        """ A method for managing the response preprocessing methods.
            args: response, voice_name
            returns: none
        """
        # Clear VRAM cache
        torch.cuda.empty_cache()
        
        # Call Sentence Splitter
        tts_response_sentences = self.split_into_sentences(response)
        
        # Clear the directories
        self.clear_directory(self.recognize_speech_dir)
        self.clear_directory(self.generate_speech_dir)
        
        # Clear audio buffer
        self.audio_data = np.array([])
        
        # Generate and store audio
        # TODO this for loop should be done with the generate audio loop
        self.generate_play_audio_loop(tts_response_sentences)
        
        #TODO remove and replace with generate play aduio loop above but upgrade to stream through chatbot api, 
        # it can still use the storage file
        for sentence in tts_response_sentences:
            if self.is_multi_speaker:
                audio = self.tts.tts(text=sentence, speaker_wav=self.voice_name_reference_speech_path, language="en", speed=3)
            else:
                audio = self.tts.tts(text=sentence, language="en", speed=3)
            
            # Convert to numpy array and append to buffer
            audio_np = np.array(audio, dtype=np.float32)
            self.audio_data = np.append(self.audio_data, audio_np)
            
            # Play audio if needed
            if not self.speech_interrupted:
                sd.play(audio_np, 22050)
                sd.wait()
                
    # -------------------------------------------------------------------------------------------------
    def play_audio_from_file(self, filename):
        """A method for audio playback from file."""
        # Check if the file exists
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist.")
            return

        try:
            # Load the audio file
            audio_data, sample_rate = sf.read(filename)

            # Play the audio file
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Failed to play audio from file {filename}. Reason: {e}")

    # -------------------------------------------------------------------------------------------------
    def generate_audio(self, sentence, ticker):
        """ a method to generate the audio for the chatbot
            args: sentence, voice_name_path, ticker
            returns: none
        """
        print(self.colors["LIGHT_CYAN"] + "🔊 generating audio for current sentence chunk 🔊:" + self.colors["RED"])
        # if self.is_multi_speaker:
        tts_audio = self.tts.tts(text=sentence, speaker_wav=self.voice_name_reference_speech_path, language="en", speed=3)
        # else:
        #     tts_audio = self.tts.tts(text=sentence, language="en", speed=3)

        # Convert to NumPy array (adjust dtype as needed)
        tts_audio = np.array(tts_audio, dtype=np.float32)

        # Save the audio with a unique name
        filename = os.path.join(self.generate_speech_dir, f"audio_{ticker}.wav")
        sf.write(filename, tts_audio, 22050)
    
    # -------------------------------------------------------------------------------------------------
    def get_audio_data(self):
        """Get the current audio buffer for visualization"""
        return self.audio_buffer[-1024:] if len(self.audio_buffer) > 0 else np.array([])
    
    # -------------------------------------------------------------------------------------------------
    def generate_play_audio_loop(self, tts_response_sentences):
        """ a method to generate and play the audio for the chatbot
            args: tts_sentences
            returns: none
        """
        # TODO /interrupt "mode" - added Shut up Feature - during audio playback loop, interupt model and allow user to overwride 
        # chat at current audio out to stop the model from talking and input new speech. 
        # Should probably make it better though, the interrupt loop doesnt function in the nextjs frontend 
        # through the api, it instead functions in the api terminal with hotkeys.
        
        # TODO /input audio "mode" "discord" - add, if modes "spacebar pressed". or "microphone input on" or "smart whisper prompting" with 
        # speech recognized and microphone "silence prompting" all as selections. Also add 2nd arg for discord audio, transcription.
        
        # TODO /decompose "mode" - pipe in Yolo, OCR, screen/game etc data decomposition, 
        # managing what data should be sent through text to speech and what should not
        
        # TODO /cut off speech "mode" - pipe interrupt data to write conversation history and mark/explain 
        # which audio chunk the user cut the model off at with the following modalities;
        #
        # (mode1) have model explain from there (always assume they didnt hear you), or 
        # (mode2) prompt the model with the marked conversation showcasing to the llm model/agent what 
        # the user did not hear through tts, but may have read on the screen text. 
        #
        # These modalities, essentially explaining to the agent wether or not the user can read the text, 
        # or can only hear, for the application of the system. This will provide for more smooth transitions 
        # for conversation, depending on modes 1 & 2.
        
        #TODO /smart listen "feature" - smart whisper/listen podcast moderator and fact checker.
        #
        #   feature set:
        #       wake commands/name call,
        #       long form whisper and audio chunking, storing processing,
        #       presume wether or not you are being spoken to, or listening to others
        #       when others are speaking, do not interrupt them, listen and whisper rec,
        #       while building conversation history, context, links, research, then
        #       forumalate possible response or responses, and when the conversation seems like
        #       both parties have felt they have been respected in their ability
        #       to uphold their free speech, the agent can jump in and say "hey guys! on
        #       that thought maybe you should try this?" 
        #
        #   tools:
        #       feature1: "longform whisper unless namecalled" on/off 
        #       feature2: "combine research data from conversation with /decompose"
        #       feature3: "listen and jump in not based on namecall, but instead based on
        #       respecting the conversation participants, and giving equal opporunities to
        #       contribute to the evolving (podcast) conversation"

        audio_queue = queue.Queue(maxsize=2)  # Queue to hold generated audio
        interrupted = False
        generation_complete = False

        def generate_audio_thread():
            nonlocal generation_complete
            for i, sentence in enumerate(tts_response_sentences):
                if interrupted:
                    break
                self.generate_audio(sentence, i)
                audio_queue.put(i)
            generation_complete = True

        # Start the audio generation thread
        threading.Thread(target=generate_audio_thread, daemon=True).start()

        ticker = 0
        while not (generation_complete and audio_queue.empty()) and not interrupted:
            try:
                current_ticker = audio_queue.get(timeout=0.1)
                filename = os.path.join(self.generate_speech_dir, f"audio_{current_ticker}.wav")
                
                play_thread = threading.Thread(target=self.play_audio_from_file, args=(filename,))
                play_thread.start()

                while play_thread.is_alive():
                    if keyboard.is_pressed('shift+alt'):
                        sd.stop()  # Stop the currently playing audio
                        interrupted = True
                        break
                    time.sleep(0.1)  # Small sleep to prevent busy-waiting

                play_thread.join()
                ticker += 1

            except queue.Empty:
                continue  # If queue is empty, continue waiting

        if interrupted:
            print(self.colors["BRIGHT_YELLOW"] + "Speech interrupted by user." + self.colors["RED"])
            self.clear_remaining_audio_files(ticker, len(tts_response_sentences))
            
    # -------------------------------------------------------------------------------------------------
    def interrupt_generation(self):
        """A method to interrupt the ongoing speech generation."""
        self.audio_queue.queue.clear()  # Clear the audio queue to stop any further audio processing
        sd.stop()  # Stop any currently playing audio
        print("Speech generation interrupted.")
        
    # -------------------------------------------------------------------------------------------------
    def clear_remaining_audio_files(self, start_ticker, total_sentences):
        """ a method to clear the audio cache from the current splicing session
        """
        for i in range(start_ticker, total_sentences):
            filename = os.path.join(self.generate_speech_dir, f"audio_{i}.wav")
            if os.path.exists(filename):
                os.remove(filename)

    # -------------------------------------------------------------------------------------------------
    def clear_directory(self, directory):
        """ a method for clearing the given directory
            args: directory
            returns: none
        """
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    # -------------------------------------------------------------------------------------------------
    def split_into_sentences(self, text: str) -> list[str]:
        """A method for splitting the LLAMA response into sentences.
        Args:
            text (str): The input text.
        Returns:
            list[str]: List of sentences.

            #TODO split by * * as well, for roleplay text such as *ahem*, *nervous laughter*, etc. when this
            is sent to a diffusion model such as bark or f5, the tts will process them as emotional sounds,
            as well as handling them as seperate chunks split from the rest which will increase speed.
            
            #TODO retrain c3po with shorter sentences and more even volume distribution

            #TODO maximum split must be less than 250 token
                - no endless sentences, -> blocks of 11 seconds, if more the model will speed up to fit it in 
                that space where you control multiple generations, instead split out chunks and handle properly.
        """
        # Add spaces around punctuation marks for consistent splitting
        text = " " + text + " "
        text = text.replace("\n", " ")

        # Handle common abbreviations and special cases
        text = re.sub(r"(Mr|Mrs|Ms|Dr|i\.e)\.", r"\1<prd>", text)
        text = re.sub(r"\.\.\.", r"<prd><prd><prd>", text)

        # Split on period, question mark, exclamation mark, or colon followed by optional spaces
        sentences = re.split(r"(?<=\d\.)\s+|(?<=[.!?:])\s+", text)

        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # Combine the number with its corresponding sentence
        combined_sentences = []
        i = 0
        while i < len(sentences):
            if re.match(r"^\d+\.", sentences[i]):
                combined_sentences.append(f"{sentences[i]} {sentences[i + 1]}")
                i += 2
            else:
                combined_sentences.append(sentences[i])
                i += 1

        # Ensure sentences are no longer than 250 characters
        final_sentences = []
        for sentence in combined_sentences:
            while len(sentence) > 250:
                # Find the nearest space before the 250th character
                split_index = sentence.rfind(' ', 0, 249)
                if split_index == -1:  # No space found, force split at 249
                    split_index = 249
                final_sentences.append(sentence[:split_index].strip())
                sentence = sentence[split_index:].strip()
            final_sentences.append(sentence)

        return final_sentences

