#!/bin/bash

# Define relative paths
CURRENT_DIR=$(pwd)
PARENT_DIR=$(dirname "$CURRENT_DIR")
AGENT_FILES_DIR="$CURRENT_DIR/AgentFiles"
IGNORED_TTS_DIR="$CURRENT_DIR/AgentFiles/Ignored_TTS"

# Check if Ollama is in PATH
if command -v ollama &> /dev/null; then
    echo "Ollama is already installed."
else
    # Prompt for custom Ollama installation path
    read -p "Ollama is not found. Do you want to install it in the default path (/usr/local/bin/ollama)? (Y/N): " ollama_path
    if [[ "$ollama_path" =~ ^[Yy]$ ]]; then
        ollama_path="/usr/local/bin/ollama"
    else
        read -p "Enter the custom path for Ollama installation: " ollama_path
    fi
    # Download and install Ollama
    curl -o OllamaSetup.sh https://ollama.com/download/OllamaSetup.sh
    chmod +x OllamaSetup.sh
    ./OllamaSetup.sh --silent --install --path="$ollama_path"
fi

# Check if Miniconda is in PATH
if command -v conda &> /dev/null; then
    echo "Miniconda is already installed."
else
    # Install Miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# Check if CUDA is installed
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA is already installed."
else
    # Install CUDA
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
fi

# Check if cuDNN is installed
if [ -d "/usr/local/cuda-11.8/cudnn" ]; then
    echo "cuDNN is already installed."
else
    # Install cuDNN
    wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.4.1/cudnn-11.8-linux-x64-v8.4.1.50.tgz
    tar -xzvf cudnn-11.8-linux-x64-v8.4.1.50.tgz
    sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
fi

# Create conda environment
conda create -n py311_ollama python=3.11 -y
source activate py311_ollama

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

# Integrate C++ Build Tools Installation
echo "Activating Conda environment..."
source activate py311_ollama

echo "Installing Microsoft C++ Build Tools..."
# Download the Visual Studio Build Tools installer
curl -L -o vs_buildtools.sh https://aka.ms/vs/17/release/vs_buildtools.sh

# Install the required components
chmod +x vs_buildtools.sh
sudo ./vs_buildtools.sh --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended

echo "Installation complete!"