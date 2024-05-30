class YourClass:
    def __init__(self):
        self.chat_history = []
        self.prompts = {
            1: "You are a helpful minecraft assistant...",
            2: "You are borch/phi3_speed_chat, a phi3 large language model..."
            # Add more prompts here as needed
        }

    def system_prompt_manager(self, sys_prompt_select):
        if sys_prompt_select in self.prompts:
            self.chat_history.append({"role": "system", "content": self.prompts[sys_prompt_select]})
        else:
            print("Invalid choice. Please select a valid prompt.")
        return sys_prompt_select