import requests
import json

class Chatbot:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.headers = {'Content-Type': 'application/json'}
        self.chat_history = []

    def send_prompt(self, user_input_prompt, user_input_model_select):
        self.chat_history.append({"user": "User", "message": user_input_prompt})

        data = {
            "model": f"{user_input_model_select}",
            "stream": False,
            "prompt": "\n".join(msg["message"] for msg in self.chat_history),
        }

        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))

        if response.status_code == 200:
            response_data = json.loads(response.text)
            llama_response = response_data.get("response")
            self.chat_history.append({"user": "Assistant", "message": llama_response})
            return llama_response
        else:
            return f"Error: {response.status_code} {response.text}"

    def save_to_json(self, filename):
        with open(filename, "w") as json_file:
            json.dump(self.chat_history, json_file, indent=2)

    def load_from_json(self, filename):
        with open(filename, "r") as json_file:
            self.chat_history = json.load(json_file)

if __name__ == "__main__":
    chatbot = Chatbot()

    user_input_model_select = input("PROVIDE AGENT NAME >>> ")
    while True:
        user_input_prompt = input(">>> ")
        if user_input_prompt.lower() == "/save":
            chatbot.save_to_json("chat_history.json")
            print("Chat history saved to chat_history.json")
        elif user_input_prompt.lower() == "/load":
            chatbot.load_from_json("chat_history.json")
            print("Chat history loaded from chat_history.json")
        else:
            response = chatbot.send_prompt(user_input_prompt, user_input_model_select)
            print(response)