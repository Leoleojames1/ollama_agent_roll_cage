"""
Created on Nov 2, 2022

    ReadWriteSymbolCollector.py is a python developer class for writing data to symbol table json files
    reading that data to a python dictionary. These symbol tables act similar to look-up books and are
    read in for the arguments of each function and at the perform step are transformed with the GetSymbolValue()
    method from the symbol table class to their true table value from their symbol value.

@author: lborcherding
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
        
        self.current_dir = os.getcwd()
        self.public_wand = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.ollama_mod_cage = os.path.abspath(os.path.join(self.public_wand, os.pardir))
        self.developer_tools = os.path.join(self.ollama_mod_cage, "developer_tools.json")
        
        self.current_dir = os.getcwd()
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.parent_dir = os.path.abspath(os.path.join(self.parent_dir, os.pardir))
        
        self.ignored_agents = os.path.join(self.parent_dir, "AgentFiles\\Ignored_Agents\\") 
        self.conversation_library = os.path.join(self.parent_dir, "AgentFiles\\pipeline\\conversation_library")
        self.model_git = 'D:\\CodingGit_StorageHDD\\model_git\\'

        return None
    
    def user_paths_template_combine():
        return

    def developer_tools_generate(self):
        """ a method for generating the developer_tools.txt from the given dict
        """
        dev_dict = {
            "API_KEY_TEST" : "A38ASSC3210AWKNT345",
            "current_dir" : f"{self.current_dir}",
            "public_wand" : f"{self.public_wand}",
            "ollama_mod_cage" : f"{self.ollama_mod_cage}",
            "developer_tools" : f"{self.developer_tools}"
        }

        self.WriteStorageDictJson(writeJsonPath=self.developer_tools, portionOneDict=dev_dict, portionTwoDict=None, fileName="developer_path_library.txt")

        return dev_dict
    
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

        return None

    # -------------------------------------------------------------------------------------------------
    def ReadJsonDict( self, readJsonPath):
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

        return None