# Start LLaMA server
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/c ollama serve"

# Wait for 1 second to let the server start
Start-Sleep -Seconds 1

# Activate Conda environment
# TODO ADD VARIABLE USER NAME
& 'C:\Users\' + $env:USERNAME + '\miniconda3\Scripts\activate.bat' 'C:\Users\' + $env:USERNAME + '\miniconda3\envs\py311_ollama'

$env:OLLAMA_NUM_PARALLEL = "2"
$env:OLLAMA_MAX_LOADED_MODELS = "2"
$env:OLLAMA_FLASH_ATTENTION = "1"
# $env:PYTHONPATH = $env:PYTHONPATH + ";D:\CodingGit_StorageHDD\python-p2p-network"

# Run Python script
Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/k python wizard_chatbot_class.py"
