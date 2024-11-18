import os
def map_directory(directory_path):
    def get_file_tree(directory, prefix=""): 
        result = ""
        items = sorted(os.listdir(directory))
        for i, item in enumerate(items):
            path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            result += f"{prefix}{'└── ' if is_last else '├── '}{item}\n"
            new_prefix = prefix + ("    " if is_last else "│   ")
            if os.path.isdir(path): result += get_file_tree(path, new_prefix)
        return result

    def get_file_contents(file_path):
        result = f"\n{'-'*100}\n{file_path}:\n{'-'*100}\n"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1): result += f"{i:>3} | {line}"
        except Exception as e: result += f"Error reading file: {e}"
        return result + f"\n{'-'*100}\n"

    output = "Directory Structure:\n\n" + get_file_tree(directory_path) + "\nFile Contents:"
    for root, _, files in os.walk(directory_path):
        for file in files: output += get_file_contents(os.path.join(root, file))
    return output

directory = "project/"
print(map_directory(directory))