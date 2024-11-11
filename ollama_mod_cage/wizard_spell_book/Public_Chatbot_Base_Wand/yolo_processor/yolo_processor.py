import gradio as gr
import json
import numpy as np
from PIL import ImageGrab
import cv2
import ollama
from ultralytics import YOLO
import multiprocessing
import logging
import os
import time
import queue
import threading

#TODO INTEGRATE YOLO PROCESSOR WITH OARC, ABSTRACT THE ABILITY TO INSTANCE THE YOLO, INSTANCE THE OBJECT CAPTURE, 
# GIVE IT THE ABILITY TO RUN IN THE FRONTEND BY MERGING THE YOLO PROCESSOR WITH THE OARC API

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class Vision:
    def __init__(self, model):
        self.model = model
        ensure_dir("model_view_output/")
        self.latest_frame = None
        self.latest_results = None
        self.lock = threading.Lock()
        
    def capture_screen(self):
        while True:
            screen = np.array(ImageGrab.grab())
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            with self.lock:
                self.latest_frame = screen
            time.sleep(0.01)  # Capture at ~100 FPS

    def detect_objects(self):
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        model = YOLO(self.model, task='detect')
        while True:
            with self.lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()
            
            results = model(frame)
            
            label_count = {}
            label_dict = {}
            for result in results:
                for box in result.boxes:
                    label = result.names[int(box.cls)]
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    label_count[label] = label_count.get(label, 0) + 1
                    if label not in label_dict:
                        label_dict[label] = []
                    label_dict[label].append(coords)

            with self.lock:
                self.latest_results = (label_count, label_dict)

            time.sleep(0.1)  # Detect at ~10 FPS

    def start_vision(self):
        capture_thread = threading.Thread(target=self.capture_screen)
        detect_thread = threading.Thread(target=self.detect_objects)
        capture_thread.start()
        detect_thread.start()

class Api:
    def __init__(self, vision_instance):
        self.vision = vision_instance

    def get_screen_with_boxes(self):
        with self.vision.lock:
            if self.vision.latest_frame is None or self.vision.latest_results is None:
                return np.zeros((100, 100, 3), dtype=np.uint8), {}
            
            frame = self.vision.latest_frame.copy()
            label_count, label_dict = self.vision.latest_results

        for label, positions in label_dict.items():
            for pos in positions:
                x1, y1, x2, y2 = map(int, pos)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {label_count[label]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, label_count

def chat(message):
    try:
        response = ollama.chat(model='llama3.1:8b', messages=[
            {
                'role': 'user',
                'content': message,
            },
        ])
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

def update_screen(api_instance):
    try:
        screen, labels = api_instance.get_screen_with_boxes()
        label_text = "\n".join([f"{label}: {count}" for label, count in labels.items()])
        return screen, label_text
    except Exception as e:
        logging.error(f"Error in update_screen: {str(e)}")
        return np.zeros((100, 100, 3), dtype=np.uint8), "Error occurred while updating screen"

def gradio_chat(message, history):
    response = chat(message)
    history.append((message, response))
    return "", history

# Gradio Interface
def create_gradio_interface(api_instance):
    def update_output():
        screen, label_text = update_screen(api_instance)
        return screen, label_text

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                image_output = gr.Image(label="Screen with Bounding Boxes")
            with gr.Column(scale=1):
                label_output = gr.Textbox(label="Detected Objects", lines=10)
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        msg.submit(gradio_chat, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

        demo.load(update_output, outputs=[image_output, label_output], every=0.1)

    return demo

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Add this line for Windows support
    vision_model = "Computer_Vision_1.3.0.onnx"  # Make sure this file exists
    
    vision_instance = Vision(vision_model)
    vision_instance.start_vision()
    
    api_instance = Api(vision_instance)
    
    demo = create_gradio_interface(api_instance)
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=5000)