@echo off
if not exist ollama_data mkdir ollama_data
for /f "tokens=1 delims= " %%a in ('ollama list ^| findstr /r ".*/.*:.*"') do (
    echo %%a >> ollama_data\\model_names.txt
)
