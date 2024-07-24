#!/bin/bash

# Define relative paths
CURRENT_DIR=$(pwd)
PARENT_DIR=$(dirname "$CURRENT_DIR")
AGENT_FILES_DIR="$CURRENT_DIR/AgentFiles"
IGNORED_TTS_DIR="$AGENT_FILES_DIR/Ignored_TTS"

# Prompt for installation paths
read -p "Enter the path to your Miniconda installation: " conda_path
read -p "Enter the path to your Ollama installation: " ollama_path

# Store paths in environment variables
export CONDA_PATH="$conda_path"
export OLLAMA_PATH="$ollama_path"

# Check if Ollama is installed
if [ -f "$OLLAMA_PATH/OllamaSetup.sh" ]; then
    echo "Ollama is already installed."
else
    # Download and install Ollama
    curl -o OllamaSetup.sh https://ollama.com/download/OllamaSetup.sh
    chmod +x OllamaSetup.sh
    ./OllamaSetup.sh --silent --install
fi

# Check if CUDA is installed
if ! command -v nvcc &> /dev/null; then
    # Install CUDA
    sudo sh cuda_installer.sh --silent
else
    echo "CUDA is already installed."
fi

# Check if cuDNN is installed
if [ -f "/usr/local/cuda/lib64/libcudnn.so" ]; then
    echo "cuDNN is already installed."
else
    # Install cuDNN
    sudo sh cudnn_installer.sh --silent
fi

# Check if Miniconda is installed
if ! command -v conda &> /dev/null; then
    # Install Miniconda3
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_PATH"
    export PATH="$CONDA_PATH/bin:$PATH"
else
    echo "Miniconda is already installed."
fi

# Create conda environment
conda create -n py311_ollama python=3.11 -y
source "$CONDA_PATH/bin/activate" py311_ollama

# Install pip wheels and NVIDIA pyindex
pip install nvidia-pyindex
pip install TTS
pip install SpeechRecognition

# Install requirements
pip install -r requirements.txt

# Download the XTTS Model for coqui
cd "$IGNORED_TTS_DIR"
git clone https://huggingface.co/coqui/XTTS-v2

# Install PyTorch and related packages
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Installation complete!"