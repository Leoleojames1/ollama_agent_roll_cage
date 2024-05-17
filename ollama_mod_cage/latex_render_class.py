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
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class latex_render_class:
    def __init__(self):
        self.root = tk.Tk()
        self.root.configure(bg='black')  # Set the background color of the Tkinter window to black
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.patch.set_facecolor('black')  # Set the background color of the figure to black
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.latex_response = ""
        self.queue = queue.Queue()
        self.new_latex = False  # Add a flag to indicate whether new LaTeX has come in
        self.root.after(100, self.check_queue)

        # self.scrollbar = Scrollbar(self.root, command=self.canvas.yview)
        # self.scrollable_frame = tk.Frame(self.canvas)

        # # Configure the canvas to be scrollable
        # self.scrollable_frame.bind(
        #     "<Configure>",
        #     lambda e: self.canvas.configure(
        #         scrollregion=self.canvas.bbox("all")
        #     )
        # )

    def add_latex_code(self, latex_response):
        """Add LaTeX code to the document."""
        self.new_latex = True  # Set the flag to True when new LaTeX comes in
        latex_blocks = self.parse_latex_code(latex_response)
        latex_blocks = latex_blocks.split('\n')
        # Add each unique LaTeX block to the queue separately
        for block in set(latex_blocks):  # Convert the list to a set to remove duplicates
            if block.strip():  # Skip empty strings
                self.queue.put(block)

    def matplotlib_latex_render(self, ax, latex_response, y_position):
        # Wrap the LaTeX code with $$...$$
        latex_response = "$ " + latex_response + " $"
        ax.text(0.5, y_position, r'%s' % latex_response, fontsize=21, ha='center', color='lightblue')

    def check_queue(self):
        """ a method for checking the latex formula render queue
        """
        if self.new_latex:  # Only clear the figure if new LaTeX has come in
            self.fig.clear()
            self.new_latex = False  # Reset the flag
        ax = self.fig.add_subplot(111)  # Add a new subplot for each formula
        ax.axis('off')  # Hide the axes
        ax.set_facecolor('black')  # Set the background color to black
        y_position = 0.95  # Start from the top of the subplot
        while not self.queue.empty():
            latex_response = self.queue.get()
            if latex_response.strip():  # Skip empty strings
                self.latex_response += latex_response + "\n"
                print(f"Rendering the following LaTeX code: {latex_response}")  # Print the LaTeX code
                self.matplotlib_latex_render(ax, latex_response, y_position)
                y_position -= 0.07  # Move down for the next formula
        self.canvas.draw()  # Draw the canvas once after all formulas have been added
        self.root.after(100, self.check_queue)

    def parse_latex_code(self, input_latex_prompt):
        """ a method parsing out the latex formula from the model prompt
            args: input_latex_prompt
            returns: output_parsed_latex
        """
        # Define the patterns for the LaTeX code blocks
        pattern1 = r"\\\[(.*?)\\\]"
        pattern2 = r"\\begin{align\*}(.*?)\\end{align\*}"

        # Use re.findall to find all matches for the patterns
        matches1 = re.findall(pattern1, input_latex_prompt, re.DOTALL)
        matches2 = re.findall(pattern2, input_latex_prompt, re.DOTALL)

        # Initialize an empty list to hold the parsed lines
        parsed_lines = []

        # If matches are found, extract the LaTeX code
        formulas = []
        for match in matches1 + matches2:
            # Extract the LaTeX code
            latex = match.strip()
            if latex:  # Skip empty strings
                formulas.append(latex)
        # Join all formulas into a single string, with each formula on a new line
        parsed_line = "\n".join(formulas)

        # Add the parsed line to the list of parsed lines
        parsed_lines.append(parsed_line)

        # Join all parsed lines into a single string, with each line on a new line
        return "\n".join(parsed_lines)