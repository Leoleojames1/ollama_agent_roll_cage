#!/bin/bash

# Start LLaMA server
gnome-terminal -- bash -c "ollama serve"

# Wait for 1 second to let the server start
sleep 1

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate py311_ollama

# Set environment variables
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# Run Python script
gnome-terminal -- bash -c "python wizard_chatbot_class.py"