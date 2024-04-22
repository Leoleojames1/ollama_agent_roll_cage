cd CodingGit_StorageHDD\Ollama_Custom_Mods\ollama_setup

setx OPENAI_API_KEY "DOES NOT MATTER, NO LOCAL API KEY"

curl https://localhost:11434/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer $OPENAI_API_KEY"   -d '{
    "model": "llama3",
    "messages": [
      {
        "role": "system",
        "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
      },
      {
        "role": "user",
        "content": "Compose a poem that explains the concept of recursion in programming."
      }
    ]
  }'

cmd /k