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

class custom_tkinter_manager:
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

    def open_combined_gui_2(self):
        # Create the base GUI
        base_gui = ctk.CTk()
        # Create the custom GUI
        custom_gui = ctk.CTk()
        
        # Display an image on the base GUI
        self.display_image("path_to_your_image.png", base_gui)

        # Display a LaTeX formula on the custom GUI
        self.display_latex_formula("S_n = \sum_{i=1}^{n} f(x_i^*) \Delta x", custom_gui)

        # Embed the custom GUI into the base GUI
        self.embed_gui(custom_gui, base_gui)
        # Start the tkinter main loop
        base_gui.mainloop()

    def display_image(self, image_path, gui):
        # Load the image
        image = PhotoImage(file=image_path)

        # Create a label to hold the image
        image_label = Label(gui, image=image)

        # Keep a reference to the image to prevent it from being garbage collected
        image_label.image = image

        # Add the label to the GUI
        image_label.pack()

    def display_latex_formula(self, formula, gui):
        # Create a new figure and axis
        fig, ax = plt.subplots()

        # Hide everything
        fig.patch.set_visible(False)
        ax.axis('off')

        # Add the LaTeX formula to the figure
        ax.text(0.5, 0.5, f'${formula}$', size=20, va='center', ha='center')

        # Draw the figure
        fig.canvas.draw()

        # Convert the figure to an image
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = Image.frombytes('RGB', (int(width), int(height)), fig.canvas.tostring_rgb())

        # Convert the image for use in tkinter
        tk_image = ImageTk.PhotoImage(image)

        # Create a label to hold the image
        image_label = Label(gui, image=tk_image)

        # Keep a reference to the image to prevent it from being garbage collected
        image_label.image = tk_image

        # Add the label to the GUI
        image_label.pack()

    def gui_call(self):
        manager = custom_tkinter_manager()
        manager.open_combined_gui("path_to_your_image.png", "S_n = \sum_{i=1}^{n} f(x_i^*) \Delta x")
        return