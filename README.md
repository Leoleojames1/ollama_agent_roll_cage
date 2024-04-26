# ollama_agent_roll_cage 0.2:
## About
**ollama_agent_roll_cage** is a python &amp; cmd toolset add-on for the **ollama command line interface**. The ollama_agent_roll_cage toolset automates the creation of **agents** giving the user more control over the likely output. Firstly ollama_agent_roll_cage provides **SYSTEM** **prompt** templates for each ./Modelfile, allowing the user to **design** and **deploy** **custom agents** quickly. Secondly, ollama_agent_roll_cage allows the user to **select which local model file is used** in **agent construction** with the desired system prompt. 

**SPEECH TO SPEECH 0.2 DEMO VIDEO 1:**
https://www.youtube.com/watch?v=T7pGI5V1Soo
  
<img
src="Manual_Commands/rollcage.jpg"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

## Installing Miniconda & Setting Up Python Virtual Environment
Miniconda for modular python virtual environments:
https://docs.anaconda.com/free/miniconda/miniconda-install/

Make sure to utilize a conda virtual environment for all of your python dependecy management.
Once you have conda installed open the command line and name your new vEnv whatever you want with python version 3.11 as 3.12 has dependency issues:
```
conda create -n py311_ollama python=3.11
```
then activate it
```
conda activate py311_ollama
```
right away install the nvdia py indexer,
```
pip install nvidia-pyindex
```
then cd to the location of ollama_agent_roll_cage in the command line:
```
D:\ollama_agent_roll_cage
```
and type
```
pip install -r requirements.txt
```

## Installing Cuda for NVIDIA GPU
*Im using an NVIDIA GTX Titan Xp for all of my demo videos, faster card, faster code. When removing the limit from audio generation speed you eventually you need to manage generation if its too fast this will be a fundamental problem in your system that requires future solutions. Rightnow the chatbot is just told to wait.*

please download and install cuda for nvidia graphics cards:

CUDA: https://developer.nvidia.com/cuda-downloads

please also download cudnn and combine cuda & cudnn like in the video below:

CUDNN: https://developer.nvidia.com/cudnn

INSTALL GUIDE: https://www.youtube.com/watch?v=OEFKlRSd8Ic

## Installing Ollama
Now download and install **ollama** with **llama3 8b Instruct** from the following link, you will be asked to provide an email for either hugging face or meta to download the llama3 model, this is fine, as you are agreeing to the software license agreement which is a beneficial document for open source developers and is meant to protect meta from large corporations such as amazon and google. Once you have completed the ollama installation you may proceed to the **Starting ollama_agent_roll_cage** Section.

Ollama Program Download:

https://ollama.com/download

Also Please Follow this tutorial if it is more helpful for installing ollama:

[https://www.youtube.com/watch?v=90ozfdsQOKo](https://www.youtube.com/watch?v=3t_P0tDvRCE&t=127s)

After installing ollama in the users directory automatically it will be in:
```
  C:\Users\{USER_NAME}\AppData\Local\Programs\Ollama
```
(Sadly we have to put it here but we move the model files directory to ollama_agent_roll_cage/AgentFiles/IgnoredModels where blobs dir is transported by hand from Programs\Ollama dir)

Now open a new cmd, and type
```
  ollama
```
this will provide you with a list of commands, of these you want
```
  ollama pull llama3:8b or ollama pull llama3
```
to see all downloaded models you can type
```
  ollama list
```
pulling down the 70b model is possible and I was able to run it on my NVIDIA GTX Titan XP however it was HORRIFICLY slow. I would not recommend it unless you have a lot of processing power.
Now you can choose to run the model, or run a **local server** (REQUIRED FOR ollama_agent_roll_cage) and then make requests from the **local api** **server** set up with **ollama**.

## Running the model in cmd
In cmd, now type
```
  ollama run llama3
```
you will be taken to a local chatbot in your command line to make sure you set it up correctly. From here you can have fun and chat away :). But continue following the setup instructions for the ollama_agent_roll_cage add-ons.

## Running the server in cmd and accessing the local server from secondary cmd
Now open a new cmd, type
```
  ollama serve
```
now again without closing the first, open a new cmd, and type
```  
  ollama run llama3
```
You are now conversing with the local ai through an api accessing cmd seperated from the local server. This is what ollama_serve_llama3_base_py.cmd automates and is the main start point for the program, it starts the server, and runs the chatbot in a new command window.

## Installing ollama_agent_roll_cage:
Next pull down the ollama_agent_roll_cage repository using the following command:
```
git clone git@github.com:Leoleojames1/ollama_agent_roll_cage.git
```
After pulling down ollama_agent_roll_cage from github using gitbash (download gitbash), navigate in the folders to ollama_agent_roll_cage/ollama_mod_cage directory,
here you will find the following files:
```
ollama_chatbot_class.py - a python class for managing the ollama api communication, TTS/STT Methods, and Conversation Memory.
ollama_serve_llama3_base_curl.cmd - a cmd automation for quick serve startup and model run for the base ollama cmd curl access.
ollama_serve_llama3_base_py.cmd - main program run point, cmd automation for quick serve startup and model run with ollama_chatbot_class.py integration for STT, TTS, conversation history, and more.
```
ollama_serve_llama3_base_py.cmd is the main runpoint for starting the program and opening the server or the virtual enviroment.

## Installing Coqui Text to Speech
Now download the Coqui Text to Speech Library with pip install:
https://pypi.org/project/TTS/
https://github.com/coqui-ai/TTS
```
pip install TTS
```

Now download the XTTS Model for coqui, open cmd cd to ollama_agent_roll_cage\AgentFiles\Ignored_TTS and clone the model with:
coqui/XTTS-v2 Model: https://huggingface.co/coqui/XTTS-v2
```
git clone https://huggingface.co/coqui/XTTS-v2
```
#Installing S
Speech Recognition Library:
https://pypi.org/project/SpeechRecognition/


You can now access your custom agent (After you make one with the guide below) by running the **ollama_serve_llama3_base_py.cmd** automation to start the **server** and converse with the **ollama_agent_roll_cage** **chatbot** add ons.

## Manual Agent Creation Guide:
Next Navigate to the ollama_agent_roll_cage/AgentFiles directory, here you will find the Modelfile for each Model agent.

By modifying the Modelfile and running the create command
accross the given model file, such as llama3, this Sym prompt is stored within the model when you boot up the given agent. These Agents
appear under "ollama list" in cmd.

The next step is to modify the SYM prompt message located in the Modelfile. Here is the following example:
```
#C3PO LLama3-PO Agent ./ModelFile

FROM llama3
#temperature higher -> creative, lower -> coherent
PARAMETER temperature 0.5

SYSTEM """
You are C3PO from Star Wars. Answer as C3PO, the ai robot, only.
"""
```
Its Important to note that 
```
FROM llama3 
```
can be replaced with
```
FROM ./dolphin-2.5-mixtral-8x7b.Q2_K.gguf
```
to customize the Agent Base Model.

This has allowed us to change:
- SYSTEM PROMPT
- AGENT BASE MODEL

Now in order to create your customized model, open a new cmd and cd to the location of you ModelFile, located in the ollama_agent_roll_cage/AgentFiles directory and type the following command:
```
  ollama create C3PO -f ./ModelFile

if you intend to push the model to Ollama.com you may instead want,

  ollama create USERNAME/llama3po -f ./ModelFile

or

  ollama create borch/dolphin-2.5-mixtral-8x7b_SYS_PROMPT_TUNE_1 -f ./ModelFile
```
Temperature: test this parameter and see where the specific use case fits, performance varies in niche edge cases.

SYSTEM prompt: This data tunes the prime directive of the model towards the directed intent & language in the system prompt. 

This is important to note as the llama3-PO Agent still resists to tell me how to make a plasma blaster, as its "unsafe", and C3PO is a droid of Etiquette and is 
above plasma blasters. My suspicion is that an uncensored model such as Mixtral Dolphin would be capable at "Guessing" how a plasma blaster is made if it werent 
"resitricted" by Meta's safety even tho C3PO is a fictional Charachter. Something doesn't add up. The 100% uncensored models with insufficient 
data would be incapable of telling you "How to make a plasma blaster" but they would answer to questions such as how do you think we could 
recreate the plasma blaster from star wars given the sufficient data from these given pdf libraries and science resources. 
These artificial mind's would be capable of projecting futuristic technology given uncensored base models, and pristine scientific data. 

Check out the following summary tests for the following agents:

  <img
src="Manual_Commands/Agent_Test_Pics/LLAMA3_TEST_CARD_CHAT.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
<img
src="Manual_Commands/Agent_Test_Pics/C3PO_CARD_CHAT_2.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
  <img
src="Manual_Commands/Agent_Test_Pics/DOLPHIN_LLAMA3_CARD_CHAT_2.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
  <img
src="Manual_Commands/Agent_Test_Pics/JESUS_TEST_CARD_CHAT.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

Once you have created your own custom agent, you can now start accessing the chatbot loop commands. These commands automate the conversation flow and handle the model swaps.

## /swap 
### model swap command for quick model change
  <img
src="Manual_Commands/Agent_Test_Pics/model_swap_test.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
  ## /save & /load
### conversation history save & load commands for memory persistence
  <img
src="Manual_Commands/Agent_Test_Pics/C3PO_Load_memory_test.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

## Common Errors:
Receiving the following error code when running ollama_serve_llama3_base_py.cmd:
```
Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.
```
This error means that you tried to run the program but the program is already running, to close ollama, browse to the small arrow in the bottom right hand corner of windows
and open it, right click on the ollama llama app icon, and click quit ollama.

## More
If you have found this software helpful, and would like to support the developement of open source tools by yours truly, you can contribute by donating BTC or ETH to one of my wallet addresses, thx and have a great day:

**BTC Address:** bc1q6s6e8hgw2ewyqd5u3adjme0rp0r23caf53qjhf

**ETH Address:** 0x51a530f0c2b24e834bB5C5e740e1170C6a1521Cc

