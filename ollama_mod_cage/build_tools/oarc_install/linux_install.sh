#!/bin/bash

# Define relative paths
CURRENT_DIR=$(pwd)
PARENT_DIR=$(dirname "$CURRENT_DIR")
AGENT_FILES_DIR="$CURRENT_DIR/AgentFiles"
IGNORED_TTS_DIR="$CURRENT_DIR/AgentFiles/Ignored_TTS"

# Log file
LOG_FILE="$CURRENT_DIR/linux_install.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

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
curl -L -o vs_buildtools.sh https://aka.ms/vs/17/release/vs_buildtools.sh
chmod +x vs_buildtools.sh
sudo ./vs_buildtools.sh --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
echo "Installation complete!"
log_message "Installation complete."
