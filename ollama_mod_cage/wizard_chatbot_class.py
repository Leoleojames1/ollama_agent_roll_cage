""" wizard_chatbot_class.py

    wizard_chatbot_class.py is the driver for ollama_agent_roll_cage, offering a place to define chatbot
    instances, and their toolsets, with the goal of automating an Agent workflow build list defined for 
    each instance. 
        
        This software was designed by Leo Borcherding with the intent of creating an easy to use
    ai interface for anyone, through Speech to Text and Text to Speech.

    Development for this software was started on: 4/20/2024 
    By: Leo Borcherding
        on github @ 
            leoleojames1/ollama_agent_roll_cage

"""

from ollama_chatbot_base import ollama_chatbot_base
import curses
import threading
import time

# -------------------------------------------------------------------------------------------------
class wizard_chatbot_class:
    """ 
    This class sets up the instances of chatbots and manages their interactions.
    """
    # -------------------------------------------------------------------------------------------------
    def __init__(self):
        """ 
        Initialize the wizard_chatbot_class with an empty list of chatbots, 
        a current_chatbot_index set to 0, and a threading lock.
        """
        self.chatbot = None
        self.current_chatbot_index = 0  # Initialize current_chatbot_index
        self.lock = threading.Lock()  # Create a lock

    # -------------------------------------------------------------------------------------------------
    def instantiate_ollama_chatbot_base(self):
        """ a method for Instantiating the ollama_chatbot_base class """
        self.ollama_chatbot_base_instance = ollama_chatbot_base() 

    # -------------------------------------------------------------------------------------------------
    def start_chatbot_main(self):
        """ start selected ollama_chatbot_base instance main """
        self.instantiate_ollama_chatbot_base()
        self.ollama_chatbot_base_instance.chatbot_main() 

    # # -------------------------------------------------------------------------------------------------
    # def test_start_chatbot_main(self, stdscr):
    #     """
    #     Initialize the chatbot interface and start the main loop.

    #     Args:
    #         stdscr (curses.window): The standard screen.

    #     Returns:
    #         None
    #     """
    #     if stdscr is None:
    #         raise ValueError("stdscr is None")

    #     # Set up the standard screen
    #     self.stdscr = stdscr
    #     curses.start_color()
    #     curses.init_pair(1, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    #     curses.init_pair(2, curses.COLOR_BLUE, curses.COLOR_BLACK)
    #     curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)

    #     # Enable keypad input and echo characters
    #     self.stdscr.keypad(True)
    #     self.stdscr.nodelay(True)
    #     curses.echo()

    #     # Create window
    #     max_y, max_x = self.stdscr.getmaxyx()
    #     win_height = max_y
    #     win_width = max_x - 2
    #     win_y = 0
    #     win_x = 1

    #     try:
    #         # win = curses.newwin(win_height, win_width, win_y, win_x)
    #         # pad = curses.newpad(100, win_width)
    #         self.stdscr.addstr(
    #             f"Created window and pad with size ({win_height}, {win_width}) and position ({win_y}, {win_x})\n"
    #         )
    #         self.stdscr.addstr("Refreshing screen...\n")
    #         self.stdscr.refresh()
    #         self.stdscr.addstr("Screen refreshed\n")
    #         time.sleep(10)
    #     except Exception as e:
    #         print(e)
    #         self.stdscr.addstr(f"Error creating window or pad: {e}\n")
    #         return

    #     # Initialize chatbot
    #     self.chatbot = ollama_chatbot_base()
    #     # self.chatbot = ollama_chatbot_base(0, pad, win, self.lock, self.stdscr)
    #     chatbot_thread = threading.Thread(target=self.chatbot.chatbot_main)
    #     chatbot_thread.start()

    #     self.stdscr.refresh()

# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ 
    The main loop for the ollama_chatbot_class. It uses a state machine for user command injection during command line prompting.
    All commands start with /, and are named logically.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'

    chatbot_instance = wizard_chatbot_class()
    chatbot_instance.start_chatbot_main()
    # curses.wrapper(chatbot_instance.start_chatbot_main)