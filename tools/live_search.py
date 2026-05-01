import requests
import json

class LiveSearchTool:
    """
    Live Search Tool: Enables Cogni Pro to fetch real-time information from the web.
    """
    def __init__(self):
        self.api_url = "https://en.wikipedia.org/w/api.php"

    def search(self, query):
        """Searches Wikipedia for the most relevant information."""
        print(f"   [LIVE_SEARCH] Searching for: {query}...")
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "utf8": 1,
            "srlimit": 3
        }
        
        headers = {'User-Agent': 'CogniProBot/1.0 (https://ubertrix.com; contact@ubertrix.com)'}
        try:
            response = requests.get(self.api_url, params=params, headers=headers)
            if response.status_code != 200:
                return None
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            
            if not search_results:
                return None
                
            # Fetch full content for the top result
            page_id = search_results[0]["pageid"]
            return self.fetch_page_content(page_id)
        except Exception as e:
            print(f"   [LIVE_SEARCH] Error: {e}")
            return None

    def fetch_page_content(self, page_id):
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "pageids": page_id
        }
        try:
            response = requests.get(self.api_url, params=params)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            content = pages.get(str(page_id), {}).get("extract", "")
            return content
        except:
            return None
