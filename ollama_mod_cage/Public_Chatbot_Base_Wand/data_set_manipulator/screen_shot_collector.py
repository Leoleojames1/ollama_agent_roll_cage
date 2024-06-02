""" screen_shot_collector.py
"""
import os
import glob
import pyautogui
import time

class screen_shot_collector:
    def __init__(self):
        """a method for initializing the class
        """
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.parent_dir = os.path.abspath(os.path.join(self.parent_dir, os.pardir))
        self.pipeline = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\")

    def get_screenshot(self):
        """ a method for taking a screenshot
            args: none
            returns: none
        """
        # Clear the llava_library directory
        files = glob.glob(os.path.join(self.llava_library, '*'))
        for f in files:
            os.remove(f)

        # Take a screenshot using PyAutoGUI
        user_screen = pyautogui.screenshot()

        # Create a path for the screenshot in the llava_library directory
        self.screenshot_path = os.path.join(self.llava_library, 'screenshot.png')

        # Save the screenshot to the file
        user_screen.save(self.screenshot_path)

        # Add a delay to ensure the screenshot is saved before it is read
        time.sleep(1)  # delay for 1 second
        screen_shot_flag = True
        return screen_shot_flag