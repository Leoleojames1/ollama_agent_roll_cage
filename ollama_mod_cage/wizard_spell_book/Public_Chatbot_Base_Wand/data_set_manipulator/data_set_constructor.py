""" tts_processor.py
    A class for processing the response sentences and audio generation for the ollama_chat_bot_class


    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 Phi-3-mini-4k-instruct
    python llama.cpp\convert-hf-to-gguf.py --outtype q8_0 --model-name Phi-3-mini-4k-instruct-q8_0 --outfile converted\Phi-3-mini-4k-instruct-q8_0.gguf Phi-3-mini-4k-instruct
"""
import os
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
import soundfile as sf

# -------------------------------------------------------------------------------------------------
class data_set_constructor:
    # -------------------------------------------------------------------------------------------------
    def __init__(self, pathLibrary):
        """a method for initializing the class
        """
        self.pathLibrary = pathLibrary
        self.current_dir = self.pathLibrary['current_dir']
        self.parent_dir = self.pathLibrary['parent_dir']
        self.ignored_pipeline_dir = self.pathLibrary['ignored_pipeline_dir']
        self.image_set_dir = self.pathLibrary['image_set_dir']
        self.video_set_dir = self.pathLibrary['video_set_dir']
        self.agent_files_dir = self.pathLibrary['agent_files_dir']
        
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

    # -------------------------------------------------------------------------------------------------
    def call_convert(self):
        # Example usage: D:\CodingGit_StorageHDD\Ollama_Custom_Mods\ollama_agent_roll_cage\AgentFiles\Public_Agents\C3PO\c3po_train_set_2\wav_set
        input_wav = f"{self.agent_files_dir}\Public_Agents\C3PO\c3po_train_set_2\wav_set\C3PO_long_train_1_r2_2_16.wav"
        output_wav = f"{self.agent_files_dir}\Public_Agents\C3PO\c3po_train_set_2\wav_set\C3PO_long_train_1_r2_2_16.wav"
        converter = FloatConverter(input_wav, output_wav, target_dtype="float8")
        converter.convert()
        
# -------------------------------------------------------------------------------------------------
class FloatConverter:
    # -------------------------------------------------------------------------------------------------
    def __init__(self, input_path, output_path, target_dtype="float32"):
        self.input_path = input_path
        self.output_path = output_path
        self.target_dtype = target_dtype

    # -------------------------------------------------------------------------------------------------
    def convert(self):
        try:
            # Read the float16 WAV file
            audio_data, sample_rate = sf.read(self.input_path)

            # Convert to the target dtype
            if self.target_dtype == "float32":
                converted_data = audio_data.astype(np.float32)
            elif self.target_dtype == "float8":
                converted_data = (audio_data * 127).astype(np.int8) / 127.0
            else:
                raise ValueError("Invalid target dtype. Choose 'float32' or 'float8'.")

            # Write the converted data to a new WAV file
            sf.write(self.output_path, converted_data, sample_rate)

            print(f"Conversion successful! Saved as {self.output_path}")
        except Exception as e:
            print(f"Error during conversion: {e}")