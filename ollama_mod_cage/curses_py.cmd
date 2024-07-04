@echo off

:: Start LLaMA server
@REM start cmd.exe /c "ollama serve"

:: Wait for 1 second to let the server start
@REM ping localhost -n 2 >nul

:: Activate Conda environment
@REM TODO ADD VARIABLE USER NAME
call C:\Users\%USERNAME%\miniconda3\Scripts\activate.bat C:\Users\%USERNAME%\miniconda3\envs\py311_ollama

set OLLAMA_NUM_PARALLEL=2
set OLLAMA_MAX_LOADED_MODELS=2
set OLLAMA_FLASH_ATTENTION=1
@REM set PYTHONPATH=%PYTHONPATH%;D:\CodingGit_StorageHDD\python-p2p-network

:: Run Python script
:: start cmd.exe /k "python ollama_chatbot_class.py"
start cmd.exe /k "python curse_me.py"
