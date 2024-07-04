#!/bin/bash

# Start LLaMA server
gnome-terminal -- bash -c "ollama serve"

# Wait for 1 second to let the server start
sleep 1

# Activate Conda environment
# TODO ADD VARIABLE USER NAME
source /mnt/c/Users/$USER/miniconda3/Scripts/activate /mnt/c/Users/$USER/miniconda3/envs/py311_ollama

export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Run Python script
# gnome-terminal -- bash -c "python ollama_chatbot_class.py"
gnome-terminal -- bash -c "python wizard_chatbot_class.py"