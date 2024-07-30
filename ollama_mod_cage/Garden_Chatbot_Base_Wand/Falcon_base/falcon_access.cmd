@echo off
setlocal enabledelayedexpansion

:: Activate Conda environment
call conda activate ai71
:: Run Python script
start cmd.exe /k "python falcon_api_access_class.py"