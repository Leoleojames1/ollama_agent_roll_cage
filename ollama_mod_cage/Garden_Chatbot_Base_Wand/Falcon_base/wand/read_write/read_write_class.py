import re
import os

class read_write_class:
    def __init__(self):
        self.name = "read_write_class"

    # -------------------------------------------------------------------------------------------------
    def read_json_dict(self, read_json_path):
        """ A function for reading the data from the template json file which contains the default hex header data,
        variable data values are replaced later.

        #TODO Create modular regular expression for current prompt that neeeds splicing, ask falcon to take our regex template 
        # and generate a new one for the output from the model, ie chain of though reasoning for modular regex coding agent
        Args:
            read_json_path
        Returns:
            read_file_dict
        """
        # initialize dictionary
        read_file_dict = {}

        # read json file key value pairs & store in dictionary
        with open(f"{read_json_path}", "r") as readJsonObject:

            # for each line in the json file search for 4 match groups
            for (count, line) in enumerate(readJsonObject):

                # search line in imageHeaderDict.json for 4 match groups, groups 1 & 3 are the dict keys, where as groups 2 & 4 are the dict values
                match = re.search(r'"(\$\([A-Za-z0-9\S]+\))" : "([A-Za-z0-9\S]+)"', line)

                if match:
                    # set key value pairs in readFileDict
                    read_file_dict[match.group(1)] = match.group(2)

            # close file
            readJsonObject.close()
        return read_file_dict
    
    # -------------------------------------------------------------------------------------------------
    def reformat_dict_remove_symbol_template(self, input_dict):
        """ a method to remove the $(arg.file_name) formatting from provided dictionary
        """
        output_dict = {}
        for key, value in input_dict.items():
            new_key = key.split('(')[1].split(')')[0].split('.')[0]
            output_dict[new_key] = value
        return output_dict