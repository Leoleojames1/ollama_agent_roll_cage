@echo off

:: Start LLaMA server
start cmd.exe /c "ollama serve"

:: Wait for 1 second to let the server start
ping localhost -n 2 >nul

:: Activate Conda environment
@REM TODO ADD VARIABLE USER NAME
call C:\Users\ADA\miniconda3\Scripts\activate.bat C:\Users\ADA\miniconda3\envs\py311_ollama

export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=2

:: Run Python script
:: start cmd.exe /k "python ollama_chatbot_class.py"
start cmd.exe /k "python ollama_chatbot_class.py"
