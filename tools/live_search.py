import requests
import json

class LiveSearchTool:
    """
    v6.5 Live Search Tool.
    Fetches real-time information from Wikipedia with deep content extraction.
    """
    def __init__(self):
        self.api_url = "https://en.wikipedia.org/w/api.php"

    def search(self, query):
        """
        Performs a search and returns the most relevant content snippet.
        """
        print(f"   [LIVE_SEARCH] Searching for: {query}...")
        
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "utf8": 1,
            "srlimit": 1
        }
        
        headers = {'User-Agent': 'CogniProBot/1.0 (https://ubertrix.com; contact@ubertrix.com)'}
        try:
            # 1. Search for the page
            response = requests.get(self.api_url, params=params, headers=headers)
            if response.status_code != 200:
                return None
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            
            if not search_results:
                return None
                
            page_title = search_results[0]['title']
            
            # 2. Fetch the actual content (extract)
            content_params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "titles": page_title,
                "exintro": True,
                "explaintext": True,
                "exchars": 1000
            }
            
            content_response = requests.get(self.api_url, params=content_params, headers=headers)
            content_data = content_response.json()
            pages = content_data.get("query", {}).get("pages", {})
            
            for page_id in pages:
                extract = pages[page_id].get("extract", "")
                if extract:
                    return extract
                    
            return None
        except Exception as e:
            print(f"   [LIVE_SEARCH Error] {e}")
            return None
