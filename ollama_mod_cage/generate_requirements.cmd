@echo off
setlocal enabledelayedexpansion

REM Define relative paths
set CURRENT_DIR=%cd%
set PARENT_DIR=%cd%\..
set AGENT_FILES_DIR=%cd%\AgentFiles

REM Activate Conda environment
call C:\Users\ADA\miniconda3\Scripts\activate.bat py311_ollama

REM Generate the pip install requirements file
pip freeze > requirements.txt

REM Check if requirements.txt is empty
for /f %%i in ('type requirements.txt ^| find /c /v ""') do set lines=%%i
if %lines%==0 (
    echo requirements.txt is empty. Generating again...
    pip freeze > requirements.txt
) else (
    echo requirements.txt file generated successfully!
)

pause