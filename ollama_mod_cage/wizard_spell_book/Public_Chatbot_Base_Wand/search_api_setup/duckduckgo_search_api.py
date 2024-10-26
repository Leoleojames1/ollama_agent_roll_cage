"""duckduckgo_search_api.py
"""
from duckduckgo_search import DDGS
import json

class DuckDuckGoSearchAPI:
    def __init__(self):
        self.test = 1
    
    def text_search(self, search_query):
        """A function for getting text search results from the DuckDuckGo API."""
        results = DDGS().text(f'{search_query}', region='wt-wt', safesearch='off', timelimit='y', max_results=10)
        formatted_results = self.format_results(results)
        print(f"DUCKDUCKGO TEXT SEARCH RESULTS: {json.dumps(formatted_results, indent=2)}")
        return formatted_results
    
    def image_search(self, search_query):
        """A function for getting image search results from the DuckDuckGo API."""
        results = DDGS().image(f'{search_query}', region='wt-wt', safesearch='off', timelimit='y', max_results=10)
        formatted_results = self.format_results(results)
        return formatted_results
    
    def format_results(self, results):
        """A function to format search results as a JSON grid."""
        grid = []
        for result in results:
            grid.append({
                "title": result.get("title"),
                "link": result.get("href"),
                "snippet": result.get("snippet")
            })
        return grid
    
    def deeper_search(self, search_query, desired_links):
        """A function to perform a deeper search on the desired links."""
        deeper_results = []
        for link in desired_links:
            # Perform a deeper search on each link (this is a placeholder for actual implementation)
            deeper_results.append({
                "link": link,
                "details": f"Deeper search results for {link}"
            })
        print(f"DEEPER SEARCH RESULTS: {json.dumps(deeper_results, indent=2)}")
        return deeper_results

# Example usage:
api = DuckDuckGoSearchAPI()
text_results = api.text_search("falcons and mathematics")
deeper_results = api.deeper_search("falcons and mathematics", [result["link"] for result in text_results])