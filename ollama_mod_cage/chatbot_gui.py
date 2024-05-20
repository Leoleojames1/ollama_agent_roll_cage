"""directory_manager_class.py
    #TODO Finish conversation history, agent .Modelfile, and text to speech voice reference file manager class. 
"""

import os
import re
import queue
import shutil
import threading
import subprocess
import tkinter as tk
import customtkinter as ctk
import random

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class chatbot_gui:
    def __init__(self):
        self.test = "test"

    def test(self):
        return

    def embed_gui(gui1, gui2):
        # Assuming gui1 and gui2 are instances of ctk.CTk
        # We will place gui1 inside gui2 on the right half

        # First, we need to get the dimensions of gui2
        width = gui2.winfo_screenwidth() // 2
        height = gui2.winfo_screenheight()

        # Now, we set the geometry of gui1 to fit the right half of gui2
        gui1.geometry(f'{width}x{height}+{width}+0')

        # Finally, we make sure gui1 is a transient window of gui2
        # This means it will always be on top of gui2 and move with it
        gui1.transient(gui2)

    def open_combined_gui(self):
        # Create the base GUI
        base_gui = ctk.CTk()
        # Create the custom GUI
        custom_gui = ctk.CTk()
        # Embed the custom GUI into the base GUI
        self.embed_gui(custom_gui, base_gui)
        # Start the tkinter main loop
        base_gui.mainloop()