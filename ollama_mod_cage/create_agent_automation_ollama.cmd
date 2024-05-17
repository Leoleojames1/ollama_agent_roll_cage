cd ..
cd AgentFiles
cd Ignored_Agents
cd %1
ollama create %1 -f ./ModelFile
echo %1 agent has been successfully created!