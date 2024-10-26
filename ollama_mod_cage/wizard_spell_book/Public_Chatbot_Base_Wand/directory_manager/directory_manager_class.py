"""directory_manager_class.py
    #TODO Finish conversation history, agent .Modelfile, and text to speech voice reference file manager class. 
"""

import os
import shutil

class directory_manager_class:
    def __init__(self):
        self.test = "test"

    def clear_directory(self, directory_path):
        try:
            # Remove all files in the directory
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Remove all subdirectories (non-empty) in the directory
            for subdirectory in os.listdir(directory_path):
                subdirectory_path = os.path.join(directory_path, subdirectory)
                if os.path.isdir(subdirectory_path):
                    shutil.rmtree(subdirectory_path)

            print(f"Successfully cleared everything in {directory_path}")
        except Exception as e:
            print(f"Error while clearing directory: {e}")

    def create_directory_if_not_exists(self, directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_path}' already exists.")

    def play_movie_mp4(self):
        """ a method for playing the given mp4 file from a name and directory
            args: movie_library_path, movie_name, file_constructor
            returns: waits for movie to play before returning, or returns to main loop while movie plays
        """
        return
    
    def google_search(self, prompt):
        """ a method for Google Search and Google Image Search
            args: prompt
            returns: search results
        """
        # Now let's integrate Google Search API
        google_search_url = "https://www.googleapis.com/customsearch/v1"  # Replace with the actual Google Search API endpoint
        google_search_params = {
            "key": "Your-Google-Search-API-Key",
            "cx": "Your-Custom-Search-ID",
            "q": prompt,
        }
        google_search_response = requests.get(google_search_url, params=google_search_params)
        google_search_data = google_search_response.json()

        # Similarly, integrate Google Image Search API
        google_image_search_url = "https://www.googleapis.com/customsearch/v1"  # Replace with the actual Google Image Search API endpoint
        google_image_search_params = {
            "key": "Your-Google-Image-Search-API-Key",
            "cx": "Your-Custom-Search-ID",
            "q": prompt,
            "searchType": "image",
        }
        google_image_search_response = requests.get(google_image_search_url, params=google_image_search_params)
        google_image_search_data = google_image_search_response.json()

        # Process the responses from Google APIs and integrate them into your chatbot's response mechanism
        # ...

        return google_search_data, google_image_search_data