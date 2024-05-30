"""
"""

import os

class model_write_class:
    def __init__(self):
        """a method for initializing the class
        """
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
    
    def splice_video(self):
        return