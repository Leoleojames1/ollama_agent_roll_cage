#!/bin/bash

# Define relative paths
CURRENT_DIR=$(pwd)
PARENT_DIR=$(dirname "$CURRENT_DIR")
AGENT_FILES_DIR="$CURRENT_DIR/AgentFiles"
IGNORED_TTS_DIR="$CURRENT_DIR/AgentFiles/Ignored_TTS"

# Log file
LOG_FILE="$CURRENT_DIR/windows_install.log"

# Function to log messages
log_message() {
    echo "$(date) - $1" >> "$LOG_FILE"
}

# CUDA installation
wget -O cuda_11.8.0_520.61.05_linux.run https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit >> "$LOG_FILE" 2>&1
log_message "CUDA installation completed."

# cuDNN installation
wget -O cudnn-11.8-linux-x64-v8.4.1.50.tgz https://developer.download.nvidia.com/compute/redist/cudnn/v8.4.1/cudnn-11.8-linux-x64-v8.4.1.50.tgz
tar -xzvf cudnn-11.8-linux-x64-v8.4.1.50.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Create conda environment
conda create -n py311_ollama python=3.11 -y
source activate py311_ollama

# Install required packages
pip install nvidia-pyindex TTS SpeechRecognition
pip install -r requirements.txt

# Download XTTS Model for coqui
cd "$IGNORED_TTS_DIR"
git clone https://huggingface.co/coqui/XTTS-v2

# Install PyTorch and Microsoft C++ Build Tools
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
echo "Activating Conda environment..."
source activate py311_ollama
wget -O vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe
./vs_buildtools.exe --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
echo "Installation complete!"
log_message "Installation complete."
