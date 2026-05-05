import requests
import base64
import re

class GitHubFetcher:
    """
    v1.0 GitHub Code Explorer.
    Fetches and analyzes code from public repositories.
    """
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CogniPro-Code-Agent"
        }

    def fetch_repo_code(self, repo_path, file_path=""):
        """
        Fetches code from a specific repository and file.
        Example: repo_path='Ubertrix/Cogin-AI', file_path='main.py'
        """
        print(f"   [GITHUB] Fetching from {repo_path}/{file_path}...")
        url = f"{self.base_url}/repos/{repo_path}/contents/{file_path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    # It's a directory, fetch the first few files
                    files_content = []
                    for item in data[:5]: # Limit to 5 files for learning
                        if item['type'] == 'file':
                            content = self.fetch_file_content(item['url'])
                            if content:
                                files_content.append({"name": item['name'], "content": content})
                    return files_content
                else:
                    # It's a single file
                    content = base64.b64decode(data['content']).decode('utf-8')
                    return [{"name": data['name'], "content": content}]
            return None
        except Exception as e:
            print(f"   [GITHUB Error] {e}")
            return None

    def fetch_file_content(self, file_url):
        try:
            response = requests.get(file_url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return base64.b64decode(data['content']).decode('utf-8')
            return None
        except:
            return None

    def detect_language(self, filename, content):
        """
        v1.0 Intelligent Language Detector.
        """
        ext = filename.split('.')[-1].lower() if '.' in filename else ""
        
        # Extension based detection
        mapping = {
            'py': 'Python',
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'cpp': 'C++',
            'c': 'C',
            'java': 'Java',
            'go': 'Go',
            'rs': 'Rust',
            'rb': 'Ruby',
            'php': 'PHP',
            'html': 'HTML',
            'css': 'CSS'
        }
        
        if ext in mapping:
            return mapping[ext]
            
        # Content based detection (Heuristics)
        if "def " in content or "import " in content: return "Python"
        if "function " in content or "const " in content: return "JavaScript"
        if "#include" in content: return "C++"
        if "public class " in content: return "Java"
        
        return "Unknown"
