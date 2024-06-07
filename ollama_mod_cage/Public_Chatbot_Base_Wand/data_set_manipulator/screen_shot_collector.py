""" screen_shot_collector.py
"""
import os
import glob
import pyautogui
import time
# -------------------------------------------------------------------------------------------------
class screen_shot_collector:
    # -------------------------------------------------------------------------------------------------
    def __init__(self, developer_tools_dict):
        """a method for initializing the class
        """
        self.developer_tools_dict = developer_tools_dict
        self.current_dir = self.developer_tools_dict['current_dir']
        self.parent_dir = self.developer_tools_dict['parent_dir']
        self.pipeline = self.developer_tools_dict['ignored_pipeline_dir']
    # -------------------------------------------------------------------------------------------------
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