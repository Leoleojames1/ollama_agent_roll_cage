```bat
@echo off

:: Start LLaMA server
start cmd.exe /c "ollama serve"

:: Wait for 1 second to let the server start
ping localhost -n 2 >nul

:: Run Python script
start cmd.exe /k "python ollama_chatbot_class.py"
```