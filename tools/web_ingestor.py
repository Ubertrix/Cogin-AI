import requests
from bs4 import BeautifulSoup
import re
import time

class WebIngestor:
    """
    Advanced Web Data Architect for Cogni Pro.
    Fetches, cleans, and structures data from various web sources.
    """
    def __init__(self, scraper=None):
        self.scraper = scraper
        self.headers = {
            'User-Agent': 'CogniProBot/1.0 (https://ubertrix.com; contact@ubertrix.com)'
        }

    def clean_text(self, text):
        """Advanced text cleaning and normalization."""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'[^\w\s.,!?;:()\-+*/=]', '', text)
        return text.strip()

    def search_knowledge(self, query, source="wikipedia"):
        """Search for specific knowledge using Wikipedia REST API."""
        if source == "wikipedia":
            return self._fetch_wikipedia_rest(query)
        return None

    def _fetch_wikipedia_rest(self, query):
        """Fetch page content using Wikipedia REST API (more stable)."""
        try:
            # 1. Search for the best matching title
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
            search_res = requests.get(search_url, headers=self.headers, timeout=10).json()
            if not search_res.get('query', {}).get('search'):
                return None
                
            title = search_res['query']['search'][0]['title']
            safe_title = title.replace(' ', '_')
            
            # 2. Get page summary and content via REST API
            rest_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"
            summary_res = requests.get(rest_url, headers=self.headers, timeout=10).json()
            
            summary = summary_res.get('extract', '')
            
            # For more content, we can try the mobile-sections or similar, 
            # but summary + search result snippet is a good start.
            # Let's try to get the full text via the traditional API but with better error handling
            content_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&titles={title}&explaintext=True&format=json"
            content_res = requests.get(content_url, headers=self.headers, timeout=10).json()
            pages = content_res.get('query', {}).get('pages', {})
            for page_id in pages:
                full_text = pages[page_id].get('extract', '')
                if full_text:
                    return full_text
            
            return summary
        except Exception as e:
            print(f"   [Wiki REST Error] {e}")
            return None
