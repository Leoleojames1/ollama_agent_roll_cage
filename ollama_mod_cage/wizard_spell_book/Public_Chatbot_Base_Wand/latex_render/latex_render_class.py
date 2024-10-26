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

class latex_render_class:
    def __init__(self):
        self.root = ctk.CTk()  # Use CTk instead of Tk
        self.root.configure(bg='black')  # Set the background color of the CTk window to black

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.patch.set_facecolor('black')  # Set the background color of the figure to black
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)  # Use CTk constants instead of Tk constants
        self.latex_response = ""
        self.queue = queue.Queue()
        self.new_latex = False  # Add a flag to indicate whether new LaTeX has come in
        self.root.after(100, self.check_queue)
        self.latex_color = 'lightblue'  # Initialize the LaTeX color

        # Create a button for changing the color
        self.color_button = ctk.CTkButton(self.root, text="Change Color", command=self.change_color, width=100, height=50)
        self.color_button.pack(side=ctk.BOTTOM)  # Attach the button to the bottom of the window

        self.root.minsize(500, 500)  # Set a minimum window size

    def redraw_latex(self):
        """Redraw the LaTeX output without clearing it."""
        ax = self.fig.add_subplot(111)  # Add a new subplot for each formula
        ax.axis('off')  # Hide the axes
        ax.set_facecolor('black')  # Set the background color to black
        y_position = 0.87  # Start from the top of the subplot
        for latex_response in self.latex_response.split('\n'):
            if latex_response.strip():  # Skip empty strings
                self.matplotlib_latex_render(ax, latex_response, y_position)
                y_position -= 0.22  # Increase spacing between formulas
        self.canvas.draw()  # Draw the canvas once after all formulas have been added

    def change_color(self):
        """Change the color of the LaTeX output."""
        # Generate a random RGB color with at least one value close to 1
        rgb = [random.random() for _ in range(3)]
        rgb[random.randint(0, 2)] = random.uniform(0.8, 1)  # Ensure at least one value is close to 1
        self.latex_color = tuple(rgb)  # Set the LaTeX color to the random RGB color
        self.redraw_latex()  # Redraw the LaTeX output to update the color

    def add_latex_code(self, latex_response, model_name):
        """Add LaTeX code to the document."""
        self.new_latex = True  # Set the flag to True when new LaTeX comes in
        latex_blocks = self.parse_latex_code(latex_response)
        latex_blocks = latex_blocks.split('\n')
        # Add each unique LaTeX block to the queue separately
        for block in set(latex_blocks):  # Convert the list to a set to remove duplicates
            if block.strip():  # Skip empty strings
                self.queue.put(block)
        self.root.mainloop()  # Start the GUI

    def matplotlib_latex_render(self, ax, latex_response, y_position):
        # Wrap the LaTeX code with $$...$$
        latex_response = "$ " + latex_response + " $"
        ax.text(0.5, y_position, r'%s' % latex_response, fontsize=21, ha='center', va='center', color=self.latex_color)  # Use the current LaTeX color

    def check_queue(self):
        """ a method for checking the latex formula render queue
        """
        if self.new_latex:  # Only clear the figure if new LaTeX has come in
            self.fig.clear()  # Clear the figure
            self.new_latex = False  # Reset the flag
        ax = self.fig.add_subplot(111)  # Add a new subplot for each formula
        ax.axis('off')  # Hide the axes
        ax.set_facecolor('black')  # Set the background color to black
        y_position = 0.87  # Start from the top of the subplot
        while not self.queue.empty():
            latex_response = self.queue.get()
            if latex_response.strip():  # Skip empty strings
                self.latex_response += latex_response + "\n"
                print(f"Rendering the following LaTeX code: {latex_response}")  # Print the LaTeX code
                self.matplotlib_latex_render(ax, latex_response, y_position)
                y_position -= 0.22  # Increase spacing between formulas
        self.canvas.draw()  # Draw the canvas once after all formulas have been added
        self.root.after(100, self.check_queue)

    def parse_latex_code(self, input_latex_prompt):
        """ a method parsing out the latex formula from the model prompt
            args: input_latex_prompt
            returns: output_parsed_latex
        """
        # Define the pattern for the LaTeX code blocks
        pattern = r"\\\[(.*?)\\\]"

        # Use re.findall to find all matches for the pattern
        matches = re.findall(pattern, input_latex_prompt, re.DOTALL)

        # Initialize an empty list to hold the parsed lines
        parsed_lines = []

        # If matches are found, extract the LaTeX code
        formulas = []
        for match in matches:
            # Extract the LaTeX code
            latex = match.strip()
            # Ignore if it contains \begin or \end
            if latex and not ("\\begin" in latex or "\\end" in latex):
                formulas.append(latex)
        # Join all formulas into a single string, with each formula on a new line
        parsed_line = "\n".join(formulas)

        # Add the parsed line to the list of parsed lines
        parsed_lines.append(parsed_line)

        # Join all parsed lines into a single string, with each line on a new line
        return "\n".join(parsed_lines)