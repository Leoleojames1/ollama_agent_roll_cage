"""duckduckgo_search_api.py
"""
from duckduckgo_search import DDGS

class duckduckgo_search_api:
    def __init__(self):
        self.test = 1
    
    def text_search(self, search_query):
        """a function for getting text search results from the duckduckgo api
        """
        results = DDGS().text(f'{search_query}', region='wt-wt', safesearch='off', timelimit='y', max_results=10)
        print(f"DUCKDUCKGO TEXT SEARCH RESULTS: {results}")
        return results
    
    def image_search(self, search_query):
        """a function for 
        """
        results = DDGS().image(f'{search_query}', region='wt-wt', safesearch='off', timelimit='y', max_results=10)
        return results