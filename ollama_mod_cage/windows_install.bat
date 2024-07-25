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
    if not exist OllamaSetup.exe (
        curl -o OllamaSetup.exe https://ollama.com/download/OllamaSetup.exe
    )
    start /wait OllamaSetup.exe /silent /install /path="!ollama_path!"
)

REM Check if Miniconda is in PATH
echo %PATH% | findstr /i /c:"conda" >nul
if %ERRORLEVEL% EQU 0 (
    echo Miniconda is already installed.
) else (
    REM Install Miniconda3
    if not exist Miniconda3-latest-Windows-x86_64.exe (
        curl -o Miniconda3-latest-Windows-x86_64.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    )
    start /wait Miniconda3-latest-Windows-x86_64.exe /S /InstallationType=JustMe /AddToPath=1 /RegisterPython=0
)

REM Check if CUDA is installed
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    echo CUDA is already installed.
) else (
    REM Install CUDA
    if not exist cuda_installer.exe (
        REM Replace with the actual download link for CUDA installer
        echo Downloading CUDA installer...
        curl -o cuda_installer.exe https://example.com/cuda_installer.exe
    )
    start /wait cuda_installer.exe /silent
)

REM Check if cuDNN is installed
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\cudnn" (
    echo cuDNN is already installed.
) else (
    REM Install cuDNN
    if not exist cudnn_installer.exe (
        REM Replace with the actual download link for cuDNN installer
        echo Downloading cuDNN installer...
        curl -o cudnn_installer.exe https://example.com/cudnn_installer.exe
    )
    start /wait cudnn_installer.exe /silent
)

REM Rest of your script...