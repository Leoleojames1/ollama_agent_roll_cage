#!/bin/bash

# Start the Ollama server
ollama serve &

# Run the 'phi3' task
ollama run phi3 &