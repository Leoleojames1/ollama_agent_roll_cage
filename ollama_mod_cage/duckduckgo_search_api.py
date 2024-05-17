"""duckduckgo_search_api.py
"""
from duckduckgo_search import DDGS

class duckduckgo_search_api:
    def __init__(self):
        self.test = 1
    
    def send_search_request_to_api(self, search_query):
        """a function for 
        """
        results = DDGS().text(f'{search_query}', region='wt-wt', safesearch='off', timelimit='y', max_results=10)
        return results
    

    def text(
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        backend: str = "api",
        max_results: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """DuckDuckGo text search generator. Query params: https://duckduckgo.com/params.

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m, y. Defaults to None.
            backend: api, html, lite. Defaults to api.
                api - collect data from https://duckduckgo.com,
                html - collect data from https://html.duckduckgo.com,
                lite - collect data from https://lite.duckduckgo.com.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Returns:
            List of dictionaries with search results.
        """
