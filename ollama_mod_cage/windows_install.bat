@echo off
setlocal enabledelayedexpansion

REM Define relative paths
set CURRENT_DIR=%cd%
set PARENT_DIR=%cd%\..
set AGENT_FILES_DIR=%cd%\AgentFiles
set IGNORED_TTS_DIR=%cd%\AgentFiles\Ignored_TTS

REM Prompt for installation paths
set /p conda_path="Enter the path to your Miniconda installation: "
set /p ollama_path="Enter the path to your Ollama installation: "

REM Store paths in environment variables
setx CONDA_PATH "!conda_path!"
setx OLLAMA_PATH "!ollama_path!"

REM Check if Ollama is installed
if exist "!OLLAMA_PATH!\OllamaSetup.exe" (
    echo Ollama is already installed.
) else (
    REM Download and install Ollama
    curl -o OllamaSetup.exe https://ollama.com/download/OllamaSetup.exe
    start /wait OllamaSetup.exe /silent /install
)

REM Check if CUDA is installed
where nvcc
IF %ERRORLEVEL% NEQ 0 (
    REM Install CUDA
    start /wait cuda_installer.exe /silent
) else (
    echo CUDA is already installed.
)

REM Check if cuDNN is installed
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll" (
    echo cuDNN is already installed.
) else (
    REM Install cuDNN
    start /wait cudnn_installer.exe /silent
)

REM Check if Miniconda is installed
where conda
IF %ERRORLEVEL% NEQ 0 (
    REM Install Miniconda3
    start /wait Miniconda3-latest-Windows-x86_64.exe /S /InstallationType=JustMe /AddToPath=1 /RegisterPython=0
) else (
    echo Miniconda is already installed.
)

REM Create conda environment
conda create -n py311_ollama python=3.11 -y
call !CONDA_PATH!\Scripts\activate.bat !CONDA_PATH!\envs\py311_ollama

REM Install pip wheels and NVIDIA pyindex
pip install nvidia-pyindex
pip install TTS
pip install SpeechRecognition

REM Install requirements
pip install -r requirements.txt

REM Download the XTTS Model for coqui
cd !IGNORED_TTS_DIR!
git clone https://huggingface.co/coqui/XTTS-v2

REM Install PyTorch and related packages
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo Installation complete!
pause