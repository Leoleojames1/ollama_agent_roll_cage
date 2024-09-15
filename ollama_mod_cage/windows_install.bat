@echo off

REM Define relative paths
set CURRENT_DIR=%cd%
for %%i in ("%CURRENT_DIR%") do set PARENT_DIR=%%~dpi
set AGENT_FILES_DIR=%CURRENT_DIR%\AgentFiles
set IGNORED_TTS_DIR=%CURRENT_DIR%\AgentFiles\Ignored_TTS

REM Log file
set LOG_FILE=%CURRENT_DIR%\windows_install.log

REM Function to log messages
:log_message
echo %date% %time% - %1 >> %LOG_FILE%
goto :eof

REM CUDA installation
powershell -Command "Invoke-WebRequest -Uri https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_windows.exe -OutFile cuda_11.8.0_520.61.05_windows.exe"
cuda_11.8.0_520.61.05_windows.exe -s -toolkit >> %LOG_FILE% 2>&1
call :log_message "CUDA installation completed."

REM cuDNN installation
powershell -Command "Invoke-WebRequest -Uri https://developer.download.nvidia.com/compute/redist/cudnn/v8.4.1/cudnn-11.8-windows-x64-v8.4.1.50.zip -OutFile cudnn-11.8-windows-x64-v8.4.1.50.zip"
powershell -Command "Expand-Archive -Path cudnn-11.8-windows-x64-v8.4.1.50.zip -DestinationPath ."
xcopy cuda\include\cudnn*.h "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include" /Y
xcopy cuda\lib\x64\cudnn*.dll "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin" /Y
xcopy cuda\lib\x64\cudnn*.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64" /Y

REM Create conda environment
call conda create -n py311_ollama python=3.11 -y
call conda activate py311_ollama

REM Install required packages
pip install nvidia-pyindex TTS SpeechRecognition
pip install -r requirements.txt

REM Download XTTS Model for coqui
cd /d %IGNORED_TTS_DIR%
git clone https://huggingface.co/coqui/XTTS-v2

REM Install PyTorch and Microsoft C++ Build Tools
call conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
echo Activating Conda environment...
call conda activate py311_ollama
powershell -Command "Invoke-WebRequest -Uri https://aka.ms/vs/17/release/vs_buildtools.exe -OutFile vs_buildtools.exe"
vs_buildtools.exe --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
echo Installation complete!
call :log_message "Installation complete."
