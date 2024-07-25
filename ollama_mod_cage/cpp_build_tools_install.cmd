@echo off
echo Activating Conda environment...
call conda activate your_env_name

echo Installing Microsoft C++ Build Tools...
:: Download the Visual Studio Build Tools installer
curl -L -o vs_buildtools.exe https://aka.ms/vs/17/release/vs_buildtools.exe

:: Install the required components
start /wait vs_buildtools.exe --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended

echo Installing TTS package...

:: Ensure pip is up-to-date
python -m pip install --upgrade pip

:: Install TTS package
pip install TTS

echo Installation complete!
pause