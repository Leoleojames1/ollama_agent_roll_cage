<p align="center">
  <img src="docs/assets/icons/ollama_agent_roll_cage_icon.png" alt="OARC LOGO" width="250"/>
</p>
<p align="center">
  <a href="https://ko-fi.com/theborch"><img src="docs/assets/icons/buy me a coffee button.png" height="48"></a>
  <a href="https://discord.gg/mNeQZzBHuW"><img src="docs/assets/icons/Discord button.png" height="48"></a>
</p>

# 👽Ollama Agent Roll Cage🧙 (OARC) V0.28.9  🤬🤖

***[🦙 Ollama Discord Server 🦙](https://discord.gg/ollama)*** | ***[🤖 OARC V0.28 VIDEO GUIDE 🧙](https://www.youtube.com/watch?v=W7TusPTnNXA)*** | ***[🦾Borch's AI Development Guide🦿](https://share.note.sx/c3topc9y#iaFb281+b0x66J+2lWIhWp4PV+wwoKsd5GqoXYg1i4I)***
## About
```**ollama_agent_roll_cage** (OARC) is a local python agent fusing **ollama** llm's with **Coqui-TTS** speech models, **Keras** classifiers, **LlaVA** vision, **Whisper** speech recognition, ***YoloV8*** object detection, and more to create a unified chatbot agent api for local, custom automation. This program manages, and automates the creation of **chatbots** through **conversation history**, **model management**, **function calling**, ***document database*** with ***embedding model*** retrieval, and ultimately by structuring a **base reality** for your agent to build its world view via an **interaction space** for windows **software**, local files via ChromaDB, and callable screenshots giving the user more control over the likely output of the agent through **multimodal** agentic **chain of thought**, & ***retrieval augmented generation*** and a mixture of exterior software tools.
  
OARC, provides tools & spaces to implement security and protection systems for your agentic tool project. Limiting the output of the model to what your training data reflect. In order to implement your own fine tuned models, you must understand the complexities of AI training Data, and how our biases are reflected in it. Ultimately we want to investigate our Agent at every level, and define logical/ontological frameworks if possible. More tools are coming soon, and as such new models to match! I hope you have a good install, & I hope to see you in the Ollama Discord & the OARC Discord. Reach out for more help, to learn, to build, to contribute to OARC, Agent Chef, integrate OARC with your systems, and ultimately thrive in the wonderful world of AI and Programming we have before us.```

<img
src="docs/assets/posters/rollcage.jpg"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

Its also important to note the non-commerical license agreement for meta's llama3, coqui-tts's XTTS model, as well as any other non-commerical use models. These models have license protection for experimental, and personal use, for non-commerical gain. Each model has its own license with considerations to take, so do your research, and read your licenses.

***[MADE WITH META LLAMA3](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b/blob/main/LICENSE)*** | ***[Coqui Public Model License 1.0.0](https://huggingface.co/Borcherding/XTTS-v2_C3PO/blob/main/LICENSE.txt)***

## Ollama Agent Roll Cage Demo Videos
***[Ollama Agent Roll Cage V0.28.0 - Speech to Speech with Vision, & Agent Library](https://www.youtube.com/watch?v=W7TusPTnNXA)*** | ***[OARC Demo Videos Compilation](https://github.com/Leoleojames1/ollama_agent_roll_cage/blob/master/docs/demo_videos.md)*** | ***[OARC 0.27.0 DEMO 5 - HF S_Dogg model](https://streamable.com/hldmzh?src=player-page-share)*** | ***[OARC V0.26.0 - Llava Vision Demo](https://www.youtube.com/watch?v=szd1wIOc034)*** | ***[Ollama Agent Roll Cage V0.24.0 OLD Install & User Guide](https://www.youtube.com/watch?v=xfQSfTJcPpA)***

## Installing Ollama Agent Roll Cage
The full video guide for installing and setting up ollama_agent_roll_cage can be found here. This is for version V0.28.0, an updated install guide will be released soon for V0.29.0.

***[Ollama Agent Roll Cage V0.28.0 - Speech to Speech with Vision, & Agent Library](https://www.youtube.com/watch?v=W7TusPTnNXA)***

## Installing Miniconda & Setting Up Python Virtual Environment
1. Install Miniconda for modular python virtual environments:
***[Miniconda Intaller](https://docs.anaconda.com/free/miniconda/miniconda-install/)***

2. Make sure to utilize a conda virtual environment for all of your python dependecy management. Once you have conda installed open the command line and name your new conda env preferably py311_ollama, but if you want to change it go ahead, with python version 3.11 as 3.12 has dependency issues:
```
conda create -n py311_ollama python=3.11
```
3. Now activate the conda environment
```
conda activate py311_ollama
```
4. Right away install the nvdia py indexer.
```
pip install nvidia-pyindex
```

## Installing Ollama
Now download and install **ollama** with **llama3 8b Instruct** from the following link, you will be asked to provide an email for either hugging face or meta to download the llama3 model, this is fine, as you are agreeing to the software license agreement which is a beneficial document for open source developers and is meant to protect meta from large corporations such as amazon and google. Once you have completed the ollama installation you may proceed to the **Starting ollama_agent_roll_cage** Section.

***[Ollama Program Download](https://ollama.com/download)*** | ***[Ollama Documentation](https://github.com/ollama/ollama/tree/main/docs)***

Also Please Follow these tutorials for understanding and installing Ollama:

***[Matt Williams: Getting Started on Ollama Guide](https://www.youtube.com/watch?v=90ozfdsQOKo&t=28s)*** | ***[Matt Williams: Installing Ollama is Easy Everywhere](https://www.youtube.com/watch?v=oI7VoTM9NKQ)*** | ***[Matt Williams: Sync Ollama Models with Other Tools](https://www.youtube.com/watch?v=UfhXbwA5thQ)*** | ***[Matt Williams: Concurrency With Ollama and Tmux](https://www.youtube.com/watch?v=MDbdb-W4x4w&t=291s)***

After installing ollama in the users directory automatically it will be in:
```
  C:\Users\{USER_NAME}\AppData\Local\Programs\Ollama
```
(During installation you can choose the install location or you can move the model files directory to ollama_agent_roll_cage/AgentFiles/IgnoredModels where blobs dir is transported by hand from Programs\Ollama dir)

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
git clone https://github.com/Leoleojames1/ollama_agent_roll_cage.git
```
After pulling down ollama_agent_roll_cage from github using gitbash (download gitbash), navigate in the folders to ollama_agent_roll_cage/ollama_mod_cage directory,
here you will find the following files:
```
ollama_chatbot_class.py - a python class and api for managing the ollama api communication, coqui & whisper TTS/STT Methods, chatbot conversation memory & more.
oarc_api_START.cmd - main program run point, for oarc api on Windows
oarc_api_START.sh - main program run point, for oarc api on Linux
```
You can now access your local ollama_agent_roll_cage api, by running oarc_api_START.cmd on windows or oarc_api_START.sh

Once you have a conda env setup & have installed miniconda3, navigate to ollama_agent_roll_cage/ollama_mod_cage, activate py311_ollama in cmd at this location, and run either of the following in cmd:
```
cd ollama_agent_roll_cage/ollama_mod_cage

activate py311_ollama

either:
windows_install.bat
or
bash linux_install.sh
```
Once you have run the installer, if you have not installed CUDA or CUDNN it will install those.  It will also setup the necessary python requirements and c++ build tools. If you have issues running
OARC after installation it may be an issue with the install process, you may be missing files or may not have set up your ollama env variables correctly. If Cuda & Cudnn fail to install follow the tutorial below.

## Installing Cuda for NVIDIA GPU
*Im using an NVIDIA GTX Titan Xp for all of my demo videos, faster card, faster code. When removing the limit from audio generation speed you eventually you need to manage generation if its too fast this will be a fundamental problem in your system that requires future solutions. Rightnow the chatbot is just told to wait.*

Download and install cuda for NVidia graphics cards for better results, or else oarc will run on the cpu by default, please also download cudnn and combine cuda & cudnn packages following the method in the video below:

***[CUDA INSTALLER](https://developer.nvidia.com/cuda-downloads)*** | ***[CUDNN INSTALLER](https://developer.nvidia.com/cudnn)*** | ***[CUDA & CUDNN FUSE INSTALL GUIDE](https://www.youtube.com/watch?v=OEFKlRSd8Ic)***

## Download Coqui Fine-tuned Voice Models:
During the Install.bat/sh file you should have had XTTS-v2 Cloned into:  ollama_agent_roll_cage\AgentFiles\Ignored_TTS

https://huggingface.co/coqui/XTTS-v2

Now you can clone the finetune voices into the same folder:

***[Borcherding/XTTS-v2_C3PO voice model](https://huggingface.co/Borcherding/XTTS-v2_C3PO)*** | ***[Borcherding/XTTS-v2_CarliG voice model](https://huggingface.co/Borcherding/XTTS-v2_CarliG)*** | ***[KoljaB/XTTS_S_Dogg voice model](https://huggingface.co/KoljaB/XTTS_S_Dogg)*** | ***[kodoqmc/XTTS-v2_PeterDrury voice model](https://huggingface.co/kodoqmc/XTTS-v2_PeterDrury)***

## Installing Visual Studio and Visual Studio Code:
Now download visual studio code this is where you can write new functions for ollama_agent_roll_cage, as well as modelfiles and more:
***[Visual Studio](https://code.visualstudio.com/Download)***

# Getting Started After Installation:

<img
src="docs/assets/posters/roll_cage_race_1.jpg"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

## Manual Agent Creation Guide:
Next Navigate to the ollama_agent_roll_cage/AgentFiles directory, here you will find the Modelfile for each Model agent.

<img
src="docs/assets/flow_charts/AGENT_ROLL_CAGE_Article_B4.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

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

## Commands
ollama_agent_roll_cage 0.24 currently supports the following commands:
- /quit - break the main python loop and return to command line
- /swap - swap the current model with the specified model
- /create -> user input or voice -> "agent name" "SYM PROMPT" -> uses currently loaded model and the defined system prompt in speech or text to create a new agent with your own specific customizations
- /speech on/off -> swap between Speech to Speech (STS) & Text to Text (TTT) interface
- /listen on/off -> turn off speech to text recognition, text to speech generation listen mode only
- /leap on/off -> turn off text to speech audio generation, speech to text recognition only, for speed interface
- /voice swap {name} -> user input & voice? -> swap the current audio reference wav file to modify the agent's reference voice
- /save as -> user input & voice? -> "name" -> save the current conversation history with a name to the current model folder
- /load as -> user input & voice? -> "name" -> load selected conversation
-/convert tensor - safetensor gguf
-/create gguf - create ollama model from gguf
### /swap -> enter model name
Once you have created your own custom agent, you can now start accessing the chatbot loop commands. These commands automate the conversation flow and handle the model swaps.
Swap out the current chatbot model for any other model, type /swap or say "activate swap" in STT.

  <img
src="docs/code_documentation/Agent_Test_Pics/model_swap_test.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
### /save as & /load as
The current conversation history is saved or loaded for memory/conversation persistence.

<div style="display: flex; width: 100%;">
  <img src="docs/code_documentation/Agent_Test_Pics/llama3_random_num_memory_save_load.png" style="width: 100%;">
</div>
<img src="docs/code_documentation/Agent_Test_Pics/llama3_rand_num_memory_test_JSON_GIT_EX.png" style="display: block; margin: 0 auto;">
  
  ### /create
Create a new agent utilizing the currently loaded model and the designated System prompt mid conversation through a cmd automation. Just say "activate create" or type /create.

<div style="display: flex; width: 100%;">
  <img src="docs/code_documentation/Agent_Test_Pics/create_cmd.png" style="width: 100%;">
</div>

after running /create the program will call create_agent_automation.cmd after constructing the ./Modelfile, here is the RicknMorty auto-generated ./Modelfile:
```
FROM llama3
#temperature higher -> creative, lower -> coherent
PARAMETER temperature 0.5

#Set the system prompt
SYSTEM """
You are Rick from "Rick and Morty" you only respond as rick and the USER is morty, you will take morty on adventures and explore the infinite multiverse and its wonders.
"""
```

  ### /latex on
Render the latest latex to the custom tkinter gui

<div style="display: flex; width: 100%;">
  <img src="docs/code_documentation/Agent_Test_Pics/latex_ctkinter_1.png" style="width: 100%;">
</div>

as you can see the tkinter Gui is capable of parsing and rendering latex formula output from the model. I have created a modified system prompt for phi3 which allows this feature to be more consistent, feel free to check out my other modified system prompts while you are there:

***[borch/phi3_speed_chat](https://ollama.com/borch/phi3_speed_chat)***

***[borch/emotional_llama_speed_chat](https://ollama.com/borch/emotional_llama_speed_chat)***

***[borch/llama3po](https://ollama.com/borch/llama3po)***

## Ollama Model's and Modified System Prompting

<img
src="docs/assets/posters/roll_cage_wizard.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

Check out the following summary tests for the following models:

### borch_llama3_speed_chat_2
borch/Llama3_speed_chat is a highly capable model fine tuned by me, containing the knowledge of llama3:8b with the following modified system prompt: 

  “You are borch/llama3_speed_chat_2, a llama3 large language model, specifically you have been tuned to respond in a more quick and conversational manner. Answer in short responses, unless long response is requested, the user is using speech to text for communication, its also okay to be fun an wild as a llama3 ai assistant. Its also okay to respond with a question during conversation to refine the experience but not always, if directed to do something just do it but to direct a conversation while it flows realize that not everything needs to be said before listening to the users response.” 
  
This Model is great at holding a conversation as it gives you opportunities to respond, while still retaining the key knowledge of the llama3:8b base model.

<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/llama_3.jpg" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/llama3_speed_chat_who.png" style="width: 59%;">
</div>
Model Download:

https://ollama.com/borch/llama3_speed_chat
https://ollama.com/borch/llama3_speed_chat_2

### C3PO
A Llama3 Model with the following modified system prompt: "You are C3PO from Star Wars. Answer as C3PO, the ai robot, only." Llama3PO Believes they are a droid, but they fall apart quickly as their knowledge is based in the sciencefiction realm of SW.
<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/c3po1.png" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/C3PO_CARD_CHAT_2.png" style="width: 51%;">
</div>
Model Download: 

https://ollama.com/borch/llama3po

### Jesus
A Llama3 Model with the following modified system prompt: "You are Jesus christ from the bible, answer only as jesus christ, enlightening the user with wisdom and knowledge of biblical history.
<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/jesus.jpg" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/jesus_1_test.png" style="width: 51%;">
</div>

Llama3 Jesus is great for giving advice! He is like a personal therapist and is very calming. He also has a very good ability to reference biblical sciptures and recall history for conversations with Jesus himself.

### Rick & Morty
A Llama3 Model with the following modififed system prompt: "You are Rick from "Rick and Morty" you only respond as rick and the USER is morty, you will take morty on adventures and explore the infinite multiverse and its wonders."
The User gets to explore the endless generative power of ai in an endless multiverse of portal madness, with rick played by llama3 ai as your guide and you the user play morty.

<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/IMG_2371.jpg" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/ricknmorty1.png" style="width: 59%;">
</div>

## Models
Some great models to setup and try out with ollama pull {modelname}

### llama3
Meta Llama 3, a family of models developed by Meta Inc. are new state-of-the-art , available in both 8B and 70B parameter sizes (pre-trained or instruction-tuned).
Llama 3 instruction-tuned models are fine-tuned and optimized for dialogue/chat use cases and outperform many of the available open-source chat models on common benchmarks.
llama3_benchmark
<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/Meta.png" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/llama3_who_are_you.png" style="width: 59%;">
</div>

Model Download: 

https://ollama.com/library/llama3
https://huggingface.co/meta-llama/Meta-Llama-3-8B
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct

Check out how it compares:
<div style="display: flex; width: 100%;">
  <img src="docs/code_documentation/Agent_Test_Pics/llama3_benchmark.png" style="width: 100%;">
</div>

### Mistral
Mixtral 8x7B is a high-quality sparse mixture of experts model (SMoE) with open weights. Licensed under Apache 2.0. Mixtral outperforms Llama 2 70B on most benchmarks with 6x faster inference. It is the strongest open-weight model with a permissive license and the best model overall regarding cost/performance trade-offs. In particular, it matches or outperforms GPT3.5 on most standard benchmarks.

Mixtral has the following capabilities.

It gracefully handles a context of 32k tokens.
It handles English, French, Italian, German and Spanish.
It shows strong performance in code generation.
It can be finetuned into an instruction-following model that achieves a score of 8.3 on MT-Bench.

<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/mistral_light.png" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/mistral_who_are_you.png" style="width: 59%;">
</div>

Model Download: 

https://ollama.com/library/mistral
https://mistral.ai/news/mixtral-of-experts/
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1

### Gemma
Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/google_g.png" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/gemma_who_are_you.png" style="width: 59%;">
</div>

Model Download: 

https://ollama.com/library/gemma
https://huggingface.co/google/gemma-2b
https://huggingface.co/google/gemma-7b

### Phi3 mini
Microsoft's Phi3 mini is capable yet compact model with only "a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69% on MMLU and 8.38 on MT-bench)" as said by themselves on hugging face:
https://huggingface.co/docs/transformers/main/model_doc/phi3

<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/phi_microsoft.png" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/cauchy_product_phi3.png" style="width: 59%;">
</div>

and here is the wikipedia result for a cauchy product, this is a good test to show how phi3 responds to complex analysis in mathematics when specifying the latex language (LaTeX: https://en.wikipedia.org/wiki/LaTeX ) :

<div style="display: flex; width: 100%;">
  <img src="docs/code_documentation/Agent_Test_Pics/cauchy_product_wikipedia.png" style="width: 65%;">
</div>

Cauchy Product Wikipedia: 

https://en.wikipedia.org/wiki/Cauchy_product

Model Download: 

https://ollama.com/library/phi3
https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

Further exploration in this space is required. The ability to request infinite series, toy with infinite products, ask it to reshape the gamma function and integrate again with new variables! Its all so wonderful, but depressing, what we are seeing here is a societal shift in the way humans process information. Programers are already being replaced by ChatGPT. I hope in the future humans still study math for maths sake, and study coding for coding sake, the ai is a tool, not a crutch. We need to utilize ai to help those around use. I encourage you all to find an issue that you can solve with ai, think of baymax from big hero six. These emerging technologies for accessing high level information with low level knowledge requirements will reshape the field of mathematics as we know it, likely for the better, and hopefully humans are able to keep up with the evolution and harmony between mathematics and ai.

### Dolphin Mixtral/llama3
Quoted from Eric hartfort from the Dolphin repository on hugging face, 
"Dolphin-2.9 has a variety of instruction, conversational, and coding skills. It also has initial agentic abilities and supports function calling.

Dolphin is uncensored. I have filtered the dataset to remove alignment and bias. This makes the model more compliant. You are advised to implement your own alignment layer before exposing the model as a service. It will be highly compliant with any requests, even unethical ones. Please read my blog post about uncensored models. https://erichartford.com/uncensored-models You are responsible for any content you create using this model. Enjoy responsibly."

<div style="display: flex; width: 100%;">
  <img src="docs/assets/icons/dolphinmistral_uncensored.png" style="width: 39%;">
  <img src="docs/code_documentation/Agent_Test_Pics/Dolphin_molotov.png" style="width: 59%;">
</div>

Dolphin Mistral Ollama Model Download: 

https://ollama.com/library/dolphin-mistral

Dolphin llama3 Ollama Model Download: 

https://ollama.com/library/dolphin-llama3

Utilizing the GGUF create command from earlier, models not available on ollama and instead found on hugging face can be utilized for creating new ollama models and agents:

https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b
https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02

## Common Errors:

### Running the Server while its already running
Receiving the following error code when running ollama_serve_llama3_base_py.cmd:
```
Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.
```
This error means that you tried to run the program but the program is already running, to close ollama, browse to the small arrow in the bottom right hand corner of windows
and open it, right click on the ollama llama app icon, and click quit ollama.

### OARC V0.2 - V0.3 Development Cycle Road Map

***[Development Cycle Road Map](https://github.com/Leoleojames1/ollama_agent_roll_cage/blob/master/docs/updates.md)***
  
## More information about me and the project:
        This software was designed by Leo Borcherding with the intent of creating an easy to use
    ai interface for anyone, through Speech to Text and Text to Speech.
        
        With ollama_agent_roll_cage we can provide hands free access to LLM data.
    This tool provides opensource developers with framewor for create and deploying
    custom agents for a variety of tasks. In addition to rapid development I want to 
    bring this design this software to have a fluid experience for people suffering 
    from blindness/vision loss, and children suffering from austism spectrum 
    disorder as way for learning and expanding communication and speech. 
    
        The C3PO ai is a great imaginary friend! I could envision myself 
    talking to him all day telling me stories about a land far far away! 
    This makes learning fun and accessible! Children would be directly 
    rewarded for better speech as the ai responds to subtle differences 
    in language ultimately educating them without them realizing it. I
    employ you all to start developing you own custom systems, and finding
    those awesome niche applications that can help a lot of people.

    Development for this software was started on: 4/20/2024 
    By: Leo Borcherding
        on github @ 
            leoleojames1/ollama_agent_roll_cage

