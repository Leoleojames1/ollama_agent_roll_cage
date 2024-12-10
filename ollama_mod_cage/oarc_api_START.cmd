@echo off
setlocal enabledelayedexpansion

:: Use the environment variable OARC_API to access the correct folder
cd /d %OARC_API%\ollama_mod_cage

:: Start LLaMA server
start cmd.exe /c "ollama serve"

:: Wait for 1 second to let the server start
ping localhost -n 2 >nul

:: Activate Conda environment
call conda activate py311_ollama

set OLLAMA_NUM_PARALLEL=2
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_FLASH_ATTENTION=1
@REM set PYTHONPATH=%PYTHONPATH%;D:\CodingGit_StorageHDD\python-p2p-network

:: Run Python script
start cmd.exe /k "python oarc_fast_api.py"

endlocal