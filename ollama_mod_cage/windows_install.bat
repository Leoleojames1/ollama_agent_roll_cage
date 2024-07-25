@echo off
setlocal enabledelayedexpansion

REM Define relative paths
set CURRENT_DIR=%cd%
set PARENT_DIR=%cd%\..
set AGENT_FILES_DIR=%cd%\AgentFiles
set IGNORED_TTS_DIR=%cd%\AgentFiles\Ignored_TTS

REM Check if Ollama is in PATH
echo %PATH% | findstr /i /c:"ollama" >nul
if %ERRORLEVEL% EQU 0 (
    echo Ollama is already installed.
) else (
    REM Prompt for custom Ollama installation path
    set /p ollama_path="Ollama is not found. Do you want to install it in the default path (C:\Program Files\Ollama)? (Y/N): "
    if /i "!ollama_path!"=="Y" (
        set ollama_path="C:\Program Files\Ollama"
    ) else (
        set /p ollama_path="Enter the custom path for Ollama installation: "
    )
    REM Download and install Ollama
    curl -o OllamaSetup.exe https://ollama.com/download/OllamaSetup.exe
    start /wait OllamaSetup.exe /silent /install /path="!ollama_path!"
)

REM Check if Miniconda is in PATH
echo %PATH% | findstr /i /c:"conda" >nul
if %ERRORLEVEL% EQU 0 (
    echo Miniconda is already installed.
) else (
    REM Install Miniconda3
    start /wait Miniconda3-latest-Windows-x86_64.exe /S /InstallationType=JustMe /AddToPath=1 /RegisterPython=0
)

REM Check if CUDA is installed
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    echo CUDA is already installed.
) else (
    REM Install CUDA
    start /wait cuda_installer.exe /silent
)

REM Check if cuDNN is installed
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\cudnn" (
    echo cuDNN is already installed.
) else (
    REM Install cuDNN
    start /wait cudnn_installer.exe /silent
)

REM Create conda environment
conda create -n py311_ollama python=3.11 -y
call conda activate py311_ollama

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

REM Integrate C++ Build Tools Installation
echo Activating Conda environment...
call conda activate py311_ollama

echo Installing Microsoft C++ Build Tools...
:: Download the Visual Studio Build Tools installer
curl -L -o vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe

:: Install the required components
start /wait vs_buildtools.exe --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended

echo Installation complete!
pause