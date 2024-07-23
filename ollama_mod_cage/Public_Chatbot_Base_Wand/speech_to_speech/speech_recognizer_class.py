import pyaudio
import speech_recognition as sr
import numpy as np

class speech_recognizer_class:
    def __init__(self, colors):
        self.colors = colors
        self.auto_speech_flag = False
        self.chunk_flag = False
        self.listen_flag = False
        self.wake_word = "Yo Jaime"
        self.recognizer = sr.Recognizer()
        self.use_wake_commands = False

    def get_audio(self):
        """ a method for getting the user audio from the microphone
            args: none
        """
        print(self.colors["OKBLUE"] + f">>> AUDIO RECORDING <<<")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []

        while self.auto_speech_flag and not self.chunk_flag:
            data = stream.read(1024)
            frames.append(data)

        print(self.colors["OKBLUE"] + f">>> AUDIO RECEIVED <<<")
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert the audio data to an AudioData object
        audio = sr.AudioData(b''.join(frames), 16000, 2)
        self.chunk_flag = False  # Set chunk_flag to False here to indicate that the audio has been received
        return audio

    def recognize_speech(self, audio):
        """ a method for calling the speech recognizer
            args: audio
            returns: speech_str
        """
        #TODO Realized current implementation calls google API, must replace with LOCAL SPEECH RECOGNITION MODEL WHISPER
        speech_str = self.recognizer.recognize_google(audio)
        print(self.colors["GREEN"] + f"<<<👂 SPEECH RECOGNIZED 👂 >>> " + self.colors["OKBLUE"] + f"{speech_str}")
        return speech_str

    def wake_words(self):
        """ A method for recognizing speech with wake word detection. """
        print(self.colors["OKBLUE"] + "Listening for wake word...")
        while True:
            audio = self.get_audio()
            
            # Convert audio data to numpy array
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            
            # Check if the audio is not silent
            if np.abs(audio_data).mean() > 500:  # Adjust this threshold as needed
                try:
                    speech_str = self.recognize_speech(audio)
                    
                    if speech_str.lower().startswith(self.wake_word.lower()):
                        # Remove the wake word from the speech string
                        actual_command = speech_str[len(self.wake_word):].strip()
                        print(self.colors["GREEN"] + f"Wake word detected! Command: {actual_command}")
                        return actual_command
                    else:
                        print(self.colors["YELLOW"] + "Speech detected, but wake word not found.")
                except sr.UnknownValueError:
                    print(self.colors["YELLOW"] + "Speech was unintelligible.")
                except sr.RequestError as e:
                    print(self.colors["RED"] + f"Could not request results; {e}")
            else:
                print(self.colors["BLUE"] + "Silence detected, continuing to listen...")

    def auto_speech_set(self, flag):
        self.auto_speech_flag = flag
        print(f"auto_speech_flag FLAG STATE: {self.auto_speech_flag}")

    def chunk_speech(self, flag):
        self.chunk_flag = flag

    def interrupt_speech(self):
        self.auto_speech_flag = False
        self.chunk_flag = False
        
    def toggle_wake_commands(self):
        self.use_wake_commands = not self.use_wake_commands
        print(f"Wake commands {'enabled' if self.use_wake_commands else 'disabled'}")
        
    # -------------------------------------------------------------------------------------------------
    def get_discord_audio_and_seperate_whisper(self):
        """ a method to get the system audio from discord, spotify, youtube, etc, to be recognized by
            the speech to text model
            args: none
            returns: none
        """
        test = "test"
        #TODO grab audio clip, and transcibe to text for input, also add transcibe and save to text json
        # also add get audio and store as audio for training and add record clone for user mic to train xtts 
