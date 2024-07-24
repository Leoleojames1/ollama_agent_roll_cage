# Initialize Conda for PowerShell
& conda init powershell

# Start LLaMA server
Start-Process "cmd.exe" -ArgumentList "/c ollama serve"

# Wait for 1 second to let the server start
Start-Sleep -Seconds 1

# Activate Conda environment
& conda activate py311_ollama

# Set environment variables
$env:OLLAMA_NUM_PARALLEL = 2
$env:OLLAMA_MAX_LOADED_MODELS = 2
$env:OLLAMA_FLASH_ATTENTION = 1
# $env:PYTHONPATH += ";D:\CodingGit_StorageHDD\python-p2p-network"

# Run Python script
Start-Process "cmd.exe" -ArgumentList "/k python wizard_chatbot_class.py"