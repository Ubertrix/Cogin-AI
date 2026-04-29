import json
import os

# Define JSON file path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(BASE_DIR, "registry", "linguistic_knowledge.json")

# Load data from JSON
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        LINGUISTIC_DATA = data.get("linguistic_data", {})
        LINGUISTIC_CORPUS = data.get("linguistic_corpus", [])
        print(f"Linguistic Knowledge loaded from JSON: {len(LINGUISTIC_CORPUS)} corpus entries.")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading linguistic JSON: {e}")
    LINGUISTIC_DATA = {}
    LINGUISTIC_CORPUS = []
