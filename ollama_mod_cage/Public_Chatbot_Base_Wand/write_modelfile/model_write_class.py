""" model_write_class
"""

import os
from speech_to_speech import tts_processor_class as tts_processor_instance

class model_write_class:
    def __init__(self):
        """a method for initializing the class
        """
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
    
    def write_model_file(self):
        """ a method to write a model file based on user inputs
            args: none
            returns: none
        """ #TODO ADD WRITE MODEL FILE CLASS
        # collect agent data with text input
        self.user_create_agent_name = input(WARNING + "<<< PROVIDE SAFETENSOR OR GGUF NAME (WITH .gguf or .safetensors) >>> " + OKBLUE)
        user_input_temperature = input(WARNING + "<<< PROVIDE NEW AGENT TEMPERATURE (0.1 - 5.0) >>> " + OKBLUE)
        # system_prompt = input(WHITE + "<<< PROVIDE SYSTEM PROMPT >>> " + OKBLUE)

        model_create_dir = os.path.join(self.ignored_agents, f"{self.user_create_agent_name}")
        model_create_file = os.path.join(self.ignored_agents, f"{self.user_create_agent_name}\\modelfile")

        try:
            # Create the new directory
            os.makedirs(model_create_dir, exist_ok=True)
            # Get current model template data
            self.ollama_show_template()
            # Create the text file
            # f.write(f"\n#Set the system prompt\n")
            # f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
            with open(model_create_file, 'w') as f:
                f.write(f"FROM {self.user_create_agent_name}\n")
                f.write(f"#temperature higher -> creative, lower -> coherent\n")
                f.write(f"PARAMETER temperature {user_input_temperature}\n")
                f.write(f"TEMPLATE \"\"\"\n{self.template}\n\"\"\"\n")
            return
        except Exception as e:
            return f"Error creating directory or text file: {str(e)}"
        
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
                mic_audio = tts_processor_instance.get_audio()
                self.user_create_agent_name = tts_processor_instance.recognize_speech(mic_audio)
                self.user_create_agent_name = tts_processor_instance.file_name_conversation_history_filter(self.user_create_agent_name)
            except sr.UnknownValueError:
                print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
            except sr.RequestError as e:
                print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

            print("Press space bar to record the new agent's temperature.")
            while not keyboard.is_pressed('space'):  # wait for space bar press
                time.sleep(0.1)
            # Now start recording
            try:
                mic_audio = tts_processor_instance.get_audio()
                user_input_temperature = tts_processor_instance.recognize_speech(mic_audio)
            except sr.UnknownValueError:
                print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
            except sr.RequestError as e:
                print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

            print("Press space bar to record the new agent's system prompt.")
            while not keyboard.is_pressed('space'):  # wait for space bar press
                time.sleep(0.1)
            # Now start recording
            try:        
                mic_audio = tts_processor_instance.get_audio()
                system_prompt = tts_processor_instance.recognize_speech(mic_audio)
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
            self.ollama_show_template()
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
            """ #TODO CREATE AGENT CLASS TO RUN THE MODEL FILES AFTER THEY HAVE BEEN CREATED
            self.model_git = model_git
            # collect agent data with stt or ttt
            if listen_flag == False:  # listen_flag is False, use speech recognition
                print("Press space bar to record the new agent's agent name.")
                while not keyboard.is_pressed('space'):  # wait for space bar press
                    time.sleep(0.1)
                # Now start recording
                try:
                    mic_audio = tts_processor_instance.get_audio()
                    self.user_create_agent_name = tts_processor_instance.recognize_speech(mic_audio)
                    self.user_create_agent_name = tts_processor_instance.file_name_conversation_history_filter(self.user_create_agent_name)
                except sr.UnknownValueError:
                    print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
                except sr.RequestError as e:
                    print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

                print("Press space bar to record the new agent's temperature.")
                while not keyboard.is_pressed('space'):  # wait for space bar press
                    time.sleep(0.1)
                # Now start recording
                try:
                    mic_audio = tts_processor_instance.get_audio()
                    user_input_temperature = tts_processor_instance.recognize_speech(mic_audio)
                except sr.UnknownValueError:
                    print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
                except sr.RequestError as e:
                    print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

                print("Press space bar to record the new agent's system prompt.")
                while not keyboard.is_pressed('space'):  # wait for space bar press
                    time.sleep(0.1)
                # Now start recording
                try:        
                    mic_audio = tts_processor_instance.get_audio()
                    system_prompt = tts_processor_instance.recognize_speech(mic_audio)
                except sr.UnknownValueError:
                    print(OKCYAN + "Google Speech Recognition could not understand audio" + OKCYAN)
                except sr.RequestError as e:
                    print(OKCYAN + "Could not request results from Google Speech Recognition service; {0}".format(e) + OKCYAN)

            elif listen_flag == True:  # listen_flag is True, use text input
                self.converted_gguf_model_name = input(WARNING + "<<< PROVIDE SAFETENSOR OR CONVERTED GGUF NAME (with EXTENTION .gguf or .safetensors) >>> " + OKBLUE)
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
                    f.write(f"FROM ./{self.converted_gguf_model_name}\n")
                    f.write(f"#temperature higher -> creative, lower -> coherent\n")
                    f.write(f"PARAMETER temperature {user_input_temperature}\n")
                    f.write(f"\n#Set the system prompt\n")
                    f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
                
                # Execute create_agent_cmd
                self.create_agent_cmd(self.user_create_agent_name, "create_agent_automation_gguf.cmd")
                return
            except Exception as e:
                return f"Error creating directory or text file: {str(e)}"