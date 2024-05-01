***[MADE WITH META LLAMA3](https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b/blob/main/LICENSE)***

***NOTE: This tool is a culmination of many different pieces of software however I want to share how amazing bing chat has been for testing and deploying new functions for each class in the program. Just thank you microsoft bing is amazing now. (and a big thanks to OpenAI for helping them do that)***

# ollama_agent_roll_cage 0.212:
## About
**ollama_agent_roll_cage** is a python &amp; cmd toolset add-on for the **ollama command line interface**. The ollama_agent_roll_cage toolset automates the creation of **agents** giving the user more control over the likely output. Firstly ollama_agent_roll_cage provides **SYSTEM** **prompt** templates for each ./Modelfile, allowing the user to **design** and **deploy** **custom agents** quickly. Secondly, ollama_agent_roll_cage allows the user to **select which local model file is used** in **agent construction** with the desired system prompt. 

**SPEECH TO SPEECH 0.2 DEMO VIDEO 1:**
https://www.youtube.com/watch?v=T7pGI5V1Soo

**/create shrek 0.212 DEMO VIDEO 2:**
https://www.youtube.com/watch?v=2IBPWCKCKjE

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


***NOTE: The llama3 Dolphin model is based on Llama-3-8b, which and is governed by META LLAMA 3 COMMUNITY LICENSE AGREEMENT:
https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b/blob/main/LICENSE***

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

Now download the XTTS Model for coqui, open command prompt and cd to ollama_agent_roll_cage\AgentFiles\Ignored_TTS and clone the model into this folder with:
coqui/XTTS-v2 Model: https://huggingface.co/coqui/XTTS-v2
```
git clone https://huggingface.co/coqui/XTTS-v2
```
## Installing py Speech Recognition (speech to text)
Speech Recognition Library:
https://pypi.org/project/SpeechRecognition/
```
pip install SpeechRecognition
```

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
Swap out the current chatbot model for any other model, type /swap or say "forward slash swap" in STT.

  <img
src="Manual_Commands/Agent_Test_Pics/model_swap_test.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
## /save & /load
The current conversation history is saved or loaded for memory/conversation persistence.

  <img
src="Manual_Commands/Agent_Test_Pics/C3PO_Load_memory_test.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">

  ## /create
Create a new agent utilizing the currently loaded model and the designated System prompt mid conversation through a cmd automation. Just say "forward slash create" or type /create.

  <img
src="Manual_Commands/Agent_Test_Pics/create_command_test1.png"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  
## Common Errors:

### Running the Server while its already running
Receiving the following error code when running ollama_serve_llama3_base_py.cmd:
```
Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.
```
This error means that you tried to run the program but the program is already running, to close ollama, browse to the small arrow in the bottom right hand corner of windows
and open it, right click on the ollama llama app icon, and click quit ollama.

## Updates 0.21, 0.22, 0.23 -> 0.3 - Development Cycle Plan - New Commands:
***UPCOMING SOON***

### Update 0.199 ***PUSHED TO GITHUB***
- /save - save current conversation to main history file
- /load - load the main conversation history file for long term intermodel conversation history keep seperate from /save as and /load as and only use to keep a long term history of your entire ollama agent base for specified history.
- /quit - break the main python loop and return to command line
- /swap - swap the current model with the specified model
  
### Update 0.21 ***PUSHED TO GITHUB***
- /create -> user input or voice -> "agent name" "SYM PROMPT" -> uses currently loaded model and the defined system prompt in speech or text to create a new agent with your own specific customizations
  
### Update 0.22
- /automatic response -> automatically response to the user if not google audio has been detetected from mic in loop, once promp response plays mic will open up again for input:
  
Speech recognized conversation will be stored as a prompt and if the prompt loades over it will prompt the model tell is to wait, and say that the audio wont be played so just take this as an input and
digest without responding. thus the next prompt which was too long will be sent afterwards, without the user being interrupted by the model. The model will only respond to the prompt by saying the key
words "roll cage namecall" inside of the prompt at any point, in which it will then respond once you stop talking and give the model enough wait time to auto respond if the message contains content and no speech
has been said for a variable amount of time to define a custom speed. (if a command is said namecall will wait for called command to execute before continuing, and the entire prompt should contain only the command)

- /save as -> user input & voice? -> "name" -> save the current conversation history with a name to the current model folder
- /load as -> user input & voice? -> "name" -> load selected conversation

- add method to manage the wait time for text to speech generation by controlling sd.wait() based on the token length of the next sentence. If tokens in next sentence are longer than current sentence, start processing next audio generation, if next sentence is not longer than current sentence, dont start text to speech generation otherwise there will be an overide
- Find a solution to storing the audio wav files seperatley such that an overide of the current audio out is not possible.
- coqui tts likely fix for seperate wave file issue: https://github.com/coqui-ai/TTS/discussions/2988
  
### Update 0.23
- /full voice on/off -> turn on/off speech to text for user input command arguments
- /speech on/off -> swap between Speech to Speech (STS) & Text to Text (TTT) interface
- /listen -> turn off speech to text recognition, text to speech generation listen mode only
- /leep -> turn off text to speech audio generation, speech to text recognition only, for speed interface
  
- /voice -> user input & voice? -> swap the current audio reference wav file to modify the agent's reference voice
- /record -> user input & voice? -> "name" -> record wav file and save to agent or to wav library
- /clone voice -> call /record, save and call /voice to swap voice instantly for instant voice clone transformation from library


### Update 0.24
- /playback -> playback any stored wav file in wav library
- /book audio -> load a book pdf or audiobook wav for playback
- /movie play "name" -> play back named movie mp4 file from library
- /music play "name" -> play back named music mp3 file from library
  
### Update 0.25
- /search {request} -> send search request to google api for context lookup
- /boost -> activate query boost utilizing secondary query boost model to improve user input requests as a preprocess for prompting the model.
- /PDF read -> user input & voice? -> "name" -> digest given pdf for context reference
- /PDF list -> list all pdfs stored in agent library

### Update 0.26  
- /generate image -> "prompt" -> generate image with custom LORA model
- /generate video -> "prompt" -> generate video with custom SORA model
- 
### Update 0.28
- /smart listen -> listen to the conversation between 2 people, record history, only trigger a response when the most likely human response would occur, i, e, talk short, give human like responses, yet still retain the knowledge of llama3. While 2 users converse, llama3 model learns the conversation flow, and know when stepping in for moderation, fact checking, search results, live in a heated debate where one would want to know the true nature of scientific data, historical data, language data, and all data in the moment of live conversation with agent roll cage
- /moderator -> make roll cage a conversation moderator for 2 different people having a conersation always listing and processing thoughts but never responding until "/yo llama what do you think about that" is asked after activating /moderator.
- /yo llama what do you think about that -> llama3 response for the /moderator chat history as a mediator between 2 people.
- /yo llama pull that up -> a copy of jamie from joe rogan using C3PO voice clone audio reference w/ google api search finds: youtube clips, wiki pedia google results, and explains the point, also screen shares macros with keyboard and/or google youtube wiki search browser. preferably with macro moves for opening complex task and managing operations. -> send to joe rogan and jamie? xD

## Optimization Plans: *** Updates 0.XX - UNKNOW, some likely soon ***
### Mojo - install
Download and install mojo, replace python setup with mojo for up to 35,000% efficiency increase.
### coqui text to speech - audio wave file live conversation generation
- add method to manage the wait time for text to speech generation by controlling sd.wait() based on the token length of the next sentence. If tokens in next sentence are longer than current sentence, start processing next audio generation, if next sentence is not longer than current sentence, dont start text to speech generation otherwise there will be an overide
- Find a solution to storing the audio wav files seperatley such that an overide of the current audio out is not possible.
- coqui tts likely fix for seperate wave file issue: https://github.com/coqui-ai/TTS/discussions/2988
- Fix issues with Multithreading & Multiprocessessing Pickling Error for Coqui TTS either in ollama_agent_roll_cage or in coqui TTS.
  
### sentence parser - comprehensive filter
- SYM PROMPT: Template sentence structure such as periods and end marks like <> model response </> for intelligent output formats designs specifically with ollama_agent_roll_cage in mind
- filter unique strings such as `` , also manage bullet points for 1. 2. 3. 4., as these are not the end of sentence periods, maybe send the response to another llm for query boost and sentence filtering

### Beyond
- add ollama_agent_roll_cage_language variant for **German, Spanish, French, Mandarin, Russian, latin? xD, arabic, hebrew, italian, hindi, japanese, portugeuse,** which will include the translations down to the /voice commands and language models.
- /swap language
  
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

If you have found this software helpful, and would like to support the developement of open source tools by yours truly, you can contribute by donating BTC or ETH to one of my wallet addresses, thx and have a great day:

**BTC Address:** bc1q6s6e8hgw2ewyqd5u3adjme0rp0r23caf53qjhf

**ETH Address:** 0x51a530f0c2b24e834bB5C5e740e1170C6a1521Cc

