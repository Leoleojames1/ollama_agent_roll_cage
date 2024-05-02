@echo off
if exist ollama_data\\model_names.txt del /F /Q ollama_data\\model_names.txt
for /f "tokens=1* delims=:" %%a in ('ollama list') do (
    if "%%b" NEQ "" (
        if not "%%a"=="failed to get console mode for stdout" (
            echo %%a >> ollama_data\\model_names.txt
        )
    )
)
