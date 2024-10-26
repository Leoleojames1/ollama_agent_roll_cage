""" send_prompt_and_tools.py

    

"""

import ollama
import base64
import json
import asyncio
import json
from typing import Dict, Any, Callable

# -------------------------------------------------------------------------------------------------  
class send_prompt_and_tools_class:
# -------------------------------------------------------------------------------------------------  
    def __init__(self):
        self.test = "test"
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.define_tools()
        
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
    async def send_prompt(self, user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        # if agent selected set up system prompts for llm
        if self.agent_flag == True:
            self.chat_history.append({"role": "system", "content": self.agent_dict["LLM_SYSTEM_PROMPT"]})
            print(self.colors["OKBLUE"] + f"<<< LLM SYSTEM >>> ")
            print(self.colors["BRIGHT_YELLOW"] + f"{self.agent_dict['LLM_SYSTEM_PROMPT']}")

        if self.memory_clear == True:
            self.chat_history = []

        # append user prompt from text or speech input
        self.chat_history.append({"role": "user", "content": user_input_prompt})

        # get the llava response and append it to the chat history only if an image is provided
        if self.llava_flag is True:
            with open(f'{self.screenshot_path}', 'rb') as f:
                user_screenshot_raw2 = base64.b64encode(f.read()).decode('utf-8')
                self.user_screenshot_raw = user_screenshot_raw2

            self.llava_response = self.llava_prompt(user_input_prompt, user_screenshot_raw2, user_input_prompt, self.vision_model)

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

        # Prepare tools for the model
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': tool_name,
                    'description': tool_info['description'],
                    'parameters': tool_info['parameters'],
                }
            }
            for tool_name, tool_info in self.tools.items()
        ]

        try:
            client = ollama.AsyncClient()
            response = await client.chat(
                model=self.user_input_model_select,
                messages=self.chat_history,
                stream=True,
                tools=tools
            )

            model_response = ''
            print(self.colors["RED"] + f"<<< 🤖 {self.user_input_model_select} 🤖 >>> " + self.colors["BRIGHT_BLACK"], end='', flush=True)
            
            async for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    model_response += content
                    print(content, end='', flush=True)
                
                # Handle tool calls
                if 'message' in chunk and 'tool_calls' in chunk['message']:
                    for tool_call in chunk['message']['tool_calls']:
                        tool_name = tool_call['function']['name']
                        if tool_name in self.tools:
                            args = json.loads(tool_call['function']['arguments'])
                            function_response = self.tools[tool_name]['function'](**args)
                            self.chat_history.append({
                                'role': 'tool',
                                'content': function_response,
                            })
                            print(f"\nFunction response: {function_response}")
                    
                    # Get final response after function call
                    final_response = await client.chat(
                        model=self.user_input_model_select,
                        messages=self.chat_history
                    )
                    model_response += final_response['message']['content']
                    print(final_response['message']['content'], end='', flush=True)

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
    def define_tools(self) -> None:
        """Define all available tools."""
        self.add_tool(
            name='get_flight_times',
            function=self.get_flight_times,
            description='Get the flight times between two cities',
            parameters={
                'type': 'object',
                'properties': {
                    'departure': {
                        'type': 'string',
                        'description': 'The departure city (airport code)',
                    },
                    'arrival': {
                        'type': 'string',
                        'description': 'The arrival city (airport code)',
                    },
                },
                'required': ['departure', 'arrival'],
            }
        )

        self.add_tool(
            name='get_weather',
            function=self.get_weather,
            description='Get the weather for a specific location',
            parameters={
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'The name of the city or location',
                    },
                },
                'required': ['location'],
            }
        )
        
    # -------------------------------------------------------------------------------------------------  
    def add_tool(self, name: str, function: Callable, description: str, parameters: Dict[str, Any]) -> None:
        """Add a new tool to the library."""
        self.tools[name] = {
            'function': function,
            'description': description,
            'parameters': parameters
        }
        
    # -------------------------------------------------------------------------------------------------  
    @staticmethod
    def get_flight_times(departure: str, arrival: str) -> str:
        flights = {
            'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
            'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
            'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
            'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
            'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
            'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
        }
        key = f'{departure}-{arrival}'.upper()
        return json.dumps(flights.get(key, {'error': f'Flight not found for route {key}'}))
    
    # -------------------------------------------------------------------------------------------------  
    @staticmethod
    def get_weather(location: str) -> str:
        weather_data = {
            'New York': {'temperature': 72, 'condition': 'Sunny'},
            'Los Angeles': {'temperature': 78, 'condition': 'Clear'},
        }
        return json.dumps(weather_data.get(location, {'error': f'Weather data not found for {location}'}))