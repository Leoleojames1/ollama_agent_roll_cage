# Start LLaMA server
Start-Process -NoNewWindow -FilePath "wsl" -ArgumentList "-d Ubuntu -- bash -c ""ollama serve"""

# Wait for 1 second to let the server start
Start-Sleep -Seconds 1

# Activate Conda environment
Invoke-Expression "$(conda shell.powershell hook | Out-String)"
conda activate py311_ollama

# Set environment variables
$env:OLLAMA_NUM_PARALLEL = 2
$env:OLLAMA_MAX_LOADED_MODELS = 2
$env:OLLAMA_FLASH_ATTENTION = 1

# Run Python script
Start-Process -NoNewWindow -FilePath "wsl" -ArgumentList "-d Ubuntu -- bash -c ""python wizard_chatbot_class.py"""
