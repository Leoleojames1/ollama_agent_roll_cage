"""
Created on Nov 2, 2022

    ReadWriteSymbolCollector.py is a python developer class for writing data to symbol table json files
    reading that data to a python dictionary. These symbol tables act similar to look-up books and are
    read in for the arguments of each function and at the perform step are transformed with the GetSymbolValue()
    method from the symbol table class to their true table value from their symbol value.

created on: Nov 2, 2022
by @LeoBorcherding
"""

import re
import os

# -------------------------------------------------------------------------------------------------
class read_write_symbol_collector:
    '''
    A class capable of reading & writing symbol table json files when called with various modes.
    '''

    def __init__(self):
        """
        Default Constructor
        """

        self.url = "http://localhost:11434/api/chat"
        self.model_git_dir = 'D:\\CodingGit_StorageHDD\\model_git\\' #TODO GET FROM developer_custom.json

        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))

        self.public_wand_dir = os.path.join(self.current_dir, "Public_Chatbot_Base_Wand")
        self.ignored_wand_dir = os.path.join(self.current_dir, "Public_Chatbot_Base_Wand")
        self.ollama_mod_cage_dir = os.path.abspath(os.path.join(self.public_wand_dir, os.pardir))

        self.developer_tools_dir = os.path.join(self.current_dir, "developer_tools.json")
        self.developer_custom_dir = os.path.join(self.current_dir, "developer_custom.json")

        self.agent_files_dir = os.path.join(self.parent_dir, "AgentFiles") 
        self.ignored_agents_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_Agents") 

        self.ignored_pipeline_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_pipeline") 
        self.llava_library_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_pipeline\\llava_library")
        self.conversation_library_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_pipeline\\conversation_library")

        self.image_dir = os.path.join(self.ignored_pipeline_dir, "data_constructor\\image_set")
        self.video_dir = os.path.join(self.ignored_pipeline_dir, "data_constructor\\video_set")

        self.speech_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_pipeline\\speech_library")
        self.recognize_speech_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_pipeline\\speech_library\\recognize_speech")
        self.generate_speech_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_pipeline\\speech_library\\generate_speech")
        self.tts_voice_ref_wav_pack_path_dir = os.path.join(self.parent_dir, "AgentFiles\\Ignored_pipeline\\public_speech\\Public_Voice_Reference_Pack")

        # TODO if the developer tools file exists
        if hasattr(self, self.developer_tools_dir):
            self.developer_tools_dict = read_write_symbol_collector.read_developer_tools_json()

    # -------------------------------------------------------------------------------------------------
    def developer_tools_generate(self):
        """ a method for generating the developer_tools.txt from the given dict
        """

        program_paths = {
            "current_dir" : f"{self.current_dir}",
            "parent_dir" : f"{self.parent_dir}",
            "agent_files_dir" : f"{self.agent_files_dir}",
            "ignored_agents_dir" : f"{self.ignored_agents_dir}",
            "ignored_pipeline_dir" : f"{self.ignored_pipeline_dir}",
            "llava_library_dir" : f"{self.llava_library_dir}",
            "ollama_mod_cage_dir" : f"{self.ollama_mod_cage_dir}",
            "public_wand_dir" : f"{self.public_wand_dir}",
            "ignored_wand_dir" : f"{self.ignored_wand_dir}",
            "developer_tools_dir" : f"{self.developer_tools_dir}",
            "developer_custom_dir" : f"{self.developer_custom_dir}",
            "conversation_library_dir" : f"{self.conversation_library_dir}",
            "image_dir" : f"{self.image_dir}",
            "video_dir" : f"{self.video_dir}",
            "speech_dir" : f"{self.speech_dir}",
            "recognize_speech_dir" : f"{self.recognize_speech_dir}",
            "generate_speech_dir" : f"{self.generate_speech_dir}",
            "tts_voice_ref_wav_pack_path_dir" : f"{self.tts_voice_ref_wav_pack_path_dir}",
        }

        # WriteStorageDictJson requires 2 dictionary args, if 1 is empty it will just write 1
        developer_custom_dict = self.ReadJsonDict(self.developer_custom_dir)
        developer_custom_dict = self.reformat_dict_remove_symbol_template(developer_custom_dict)
        
        # write dictionary data to json
        self.WriteStorageDictJson(writeJsonPath=self.developer_tools_dir, \
            portionOneDict=program_paths, portionTwoDict=developer_custom_dict, fileName="developer_tools.txt")

    # -------------------------------------------------------------------------------------------------
    def read_developer_tools_json(self):
        developer_dict = self.ReadJsonDict(self.developer_tools_dir)
        developer_dict_filtered = self.reformat_dict_remove_symbol_template(developer_dict)
        return developer_dict_filtered
    
    # -------------------------------------------------------------------------------------------------
    def ReadJsonDict(self, readJsonPath):
        """ A function for reading the data from the template json file which contains the default hex header data,
        variable data values are replaced later.

        #TODO Create modular regular expression based on the character occurence rate, ex:
            def readRandomFile (randomFilePath):
                with open randomFilePath :
                    for line in randomFilePath count all ascii characters, for highest occuring, or most
        Args:
            writeJsonPath
        Returns:
            readFileDict
        """
        # initialize dictionary
        readFileDict = {}

        # read json file key value pairs & store in dictionary
        with open(f"{readJsonPath}", "r") as readJsonObject:

            # for each line in the json file search for 4 match groups
            for (count, line) in enumerate(readJsonObject):

                # search line in imageHeaderDict.json for 4 match groups, groups 1 & 3 are the dict keys, where as groups 2 & 4 are the dict values
                match = re.search(r'"(\$\([A-Za-z0-9\S]+\))" : "([A-Za-z0-9\S]+)"', line)
                # match = re.search(r'"(\$\([A-Za-z0-9_]+))" : "([A-Za-z0-9_]+)"', line)
                # if match is found store groups in readFileDict
                if match:
                    # set key value pairs in readFileDict
                    readFileDict[match.group(1)] = match.group(2)

            # close file
            readJsonObject.close()
        return readFileDict
    
    # -------------------------------------------------------------------------------------------------
    def WriteStorageDictJson(self, writeJsonPath, portionOneDict, portionTwoDict, fileName):
        """ A Function which takes the self.__storageDict from the Perform function, and writes
        the file data to the file named StorageDict.json
        Args:
            mapFile, wizDict
        Returns:
            fileLengthDict
        """

        # Check for empty dict
        if portionOneDict == {}:
            pass
        else:
            # if not empty get last key
            lastKeyONE = list(portionOneDict.keys())[-1]

        # Check for empty dict
        if portionTwoDict == {}:
            pass
        else:
            # if not empty get last key
            lastKeyTWO = list(portionTwoDict.keys())[-1]

        # remove file type from name string
        fileName = fileName.split('.')[0]

        # Set Values
        keyIndexNum = -1
        columns = 1

        # Open the Json file, and write with the storageDict data
        with open( f"{writeJsonPath}", "w" ) as writeJsonObject:

            # Open Dictionary
            writeJsonObject.truncate(0)
            writeJsonObject.write("{\n")

            # for keys
            for key in portionOneDict:
                keyIndexNum += 1
                #write data in string format
                writeJsonObject.write("    ")
                writeJsonObject.write('"')
                writeJsonObject.write(f'$({key}.{fileName})')
                writeJsonObject.write('"')
                writeJsonObject.write(' : ')
                writeJsonObject.write('"')
                writeJsonObject.write(f'{portionOneDict[key]}')
                writeJsonObject.write('"')
                # skip comma for last key
                if key == lastKeyONE:
                    writeJsonObject.write(',')
                    pass
                else:
                    writeJsonObject.write(',')

                if (keyIndexNum+1) % columns == 0 :
                    writeJsonObject.write("\n")

            for key in portionTwoDict:
                keyIndexNum += 1
                #write data in string format
                writeJsonObject.write("    ")
                writeJsonObject.write('"')
                writeJsonObject.write(f'$({key}.{fileName})')
                writeJsonObject.write('"')
                writeJsonObject.write(' : ')
                writeJsonObject.write('"')
                writeJsonObject.write(f'{portionTwoDict[key]}')
                writeJsonObject.write('"')
                # skip comma for last key
                if key == lastKeyTWO:
                    pass
                else:
                    writeJsonObject.write(',')

                if (keyIndexNum+1) % columns == 0:
                    writeJsonObject.write("\n")

            # Close Dictionary
            writeJsonObject.write("\n}")

    # -------------------------------------------------------------------------------------------------
    def CombineJsonDictFiles(self, frontJsonData, backJsonData, writeJsonPath, fileName):
        """ A Function which takes the self.__storageDict from the Perform function, and writes
        the file data to the file named StorageDict.json
        Args:
            mapFile, wizDict
        Returns:
            fileLengthDict
        """
        # read json files and store data in dictionary
        frontJsonDataDict = self.ReadJsonDict(frontJsonData)
        backJsonDataDict = self.ReadJsonDict(backJsonData)

        # write front and back dict data to the given file path
        self.WriteStorageDictJson(writeJsonPath, frontJsonDataDict, backJsonDataDict, fileName)

    # -------------------------------------------------------------------------------------------------
    def reformat_dict_remove_symbol_template(self, input_dict):
        """ a method to remove the $(arg.file_name) formatting from provided dictionary
        """
        output_dict = {}
        for key, value in input_dict.items():
            new_key = key.split('(')[1].split(')')[0].split('.')[0]
            output_dict[new_key] = value
        return output_dict

    # -------------------------------------------------------------------------------------------------
    def back_slash_filter_path_dict(self):
        #TODO FINISH, REPLACE // with //// or / with // for developer tools, self. arg conversion
        for path in self.path_dict:
            for char in path:
                re.search()