@echo off
if not exist ollama_data mkdir ollama_data
if exist model_names.txt del /F /Q model_names.txt
for /f "tokens=1 delims= " %%a in ('ollama list ^| findstr /r ".*:.*"') do (
    echo %%a >> ollama_data\\model_names.txt
)