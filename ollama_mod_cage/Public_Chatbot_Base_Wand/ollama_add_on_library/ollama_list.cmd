@echo off
if exist ignored_batch_data_model_list\\model_names.txt del /F /Q ignored_batch_data_model_list\\model_names.txt
for /f "tokens=1* delims=:" %%a in ('ollama list') do (
    if "%%b" NEQ "" (
        if not "%%a"=="failed to get console mode for stdout" (
            echo %%a >> ollama_batch_data_model_list\\model_names.txt
        )
    )
)
