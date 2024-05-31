""" prompt_base_methods.py
"""

import os
import base64
import ollama

class prompt_base_methods:
    def __init__(self):
        self.current_dir = os.getcwd()

    def send_prompt(self, user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
        #TODO ADD IF MEM OFF CLEAR HISTORY
        self.chat_history = []
        self.screenshot_path = os.path.join(self.llava_library, "screenshot.png")

        #TODO ADD SYSTEM PROMP MANAGER FOR DIFFERENT MODES
        # Minecraft
        self.chat_history.append({"role": "system", "content": "You are a helpful minecraft assistant, given the provided screenshot data please direct the user immediatedly, prioritize the order in which to inform the player, hostile mobs should be avoided or terminated, danger is a top priority, but so is crafting and building, if they require help quickly guide them to a solution in real time. Please respond in a quick conversational voice, do not read off of documentation, you need to directly explain quickly and effectively whats happening, for example if there is a zombie say something like, watch out thats a Zombie hurry up and kill it or run away, they are dangerous. The recognized Objects around the perimeter are usually items, health, hunger, breath, gui elements, or status affects, please differentiate these objects in the list from 3D objects in the forward facing perspective with hills trees, mobs etc, the items are held by the player and due to the perspective take up the warped edge of the image on the sides. the sky is typically up with a sun or moon and stars, with the dirt below, there is also the nether which is a firey wasteland and cave systems with ore. Please stick to whats relevant to the current user prompt and llava data:"})
        # phi3 speed chat
        # self.chat_history.append({"role": "system", "content": "You are borch/phi3_speed_chat, a phi3 large language model, specifically you have been tuned to respond in a more quick and conversational manner, the user is using speech to text for communication, its also okay to be fun and wild as a phi3 ai assistant. Its also okay to respond with a question, if directed to do something just do it, and realize that not everything needs to be said in one shot, have a back and forth listening to the users response. If the user decides to request a latex math code output, use \[...\] instead of $$...$$ notation, if the user does not request latex, refrain from using latex unless necessary. Do not re-explain your response in a parend or bracketed note: the response... this is annoying and users dont like it."})
        
        # append user prompt
        self.chat_history.append({"role": "user", "content": user_input_prompt})

        # get the llava response and append it to the chat history only if an image is provided
        if self.llava_flag is True:
            # load the screenshot and convert it to a base64 string
            with open(f'{self.screenshot_path}', 'rb') as f:
                user_screenshot_raw2 = base64.b64encode(f.read()).decode('utf-8')
                self.user_screenshot_raw = user_screenshot_raw2
            llava_response = self.llava_prompt(user_screenshot_raw2, user_input_prompt)
            print(f"LLAVA SOURCE: {llava_response}")
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
        
    def llava_prompt(self, user_screenshot_raw2, user_input_prompt):
        """ a method for prompting the model
            args: user_input_prompt, user_input_model_select, search_google
            returns: none
        """
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