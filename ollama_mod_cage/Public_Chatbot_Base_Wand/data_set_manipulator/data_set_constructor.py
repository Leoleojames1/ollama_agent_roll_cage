""" tts_processor.py
    A class for processing the response sentences and audio generation for the ollama_chat_bot_class


    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 Phi-3-mini-4k-instruct
    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 --outfile converted\Phi-3-mini-4k-instruct-q8_0.gguf Phi-3-mini-4k-instruct
"""
import os
from moviepy.editor import VideoFileClip
from PIL import Image

# -------------------------------------------------------------------------------------------------
class data_set_constructor:
    # -------------------------------------------------------------------------------------------------
    def __init__(self, developer_tools_dict):
        """a method for initializing the class
        """
        self.developer_tools_dict = developer_tools_dict
        self.current_dir = self.developer_tools_dict['current_dir']
        self.parent_dir = self.developer_tools_dict['parent_dir']
        self.ignored_pipeline_dir = self.developer_tools_dict['ignored_pipeline_dir']
        self.image_dir = self.developer_tools_dict['image_dir']
        self.video_dir = self.developer_tools_dict['video_dir']
        
    # -------------------------------------------------------------------------------------------------
    def splice_video(self, video_path):
        # Load video
        clip = VideoFileClip(video_path)

        # Get the video name (without extension) to use in the image name
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Create a new directory for this video's images
        video_image_dir = os.path.join(self.image_dir, video_name)
        os.makedirs(video_image_dir, exist_ok=True)

        # Iterate over the duration of the clip with a step size of 'interval'
        for i in range(0, int(clip.duration), 10):
            # Get a frame at 'i' seconds
            frame = clip.get_frame(i)
            
            # Save the frame as an image
            frame_image = Image.fromarray(frame)
            frame_image.save(os.path.join(video_image_dir, f'frame_{i}.png'))

        print(f"Frames saved in directory {video_image_dir}")

    # -------------------------------------------------------------------------------------------------
    def generate_image_data(self):  # Changed interval to 30
        # Iterate over all videos in the video directory
        for video_file in os.listdir(self.video_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(self.video_dir, video_file)
                self.splice_video(video_path)  # Splice video every 'interval' seconds