import os
import numpy as np
from main import CogniPro

print("--- V3.0 Advanced Engineering Verification ---")

# 1. Sanitation Check
files = os.listdir(".")
archive_exists = os.path.isdir("archive")
scrapers_archived = not any(f.startswith("web_scraper_ingest") for f in files)
print(f"Archive Folder: {'OK' if archive_exists else 'MISSING'}")
print(f"Legacy Scrapers Cleaned: {'OK' if scrapers_archived else 'STILL IN ROOT'}")

# 2. Ingestion & Metadata Check
cp = CogniPro()
# Search for something we know was ingested with V3 branding
# The Scraper injects "Distilled Logic: " prefix.
res = cp.long_term.retrieve(query_text="Python functions", top_k=1)
if res[0] is None:
    # Try searching for the exact trigger prefix if direct search failed
    res = cp.long_term.retrieve(query_text="Distilled Logic: Python", top_k=1)

concept, vec, cat, label, data = res
print(f"Retrieved Entry: {concept} | Category: {cat}")

if data:
    weight = data.get('neuron_weight') if isinstance(data, dict) else "N/A"
    is_new = data.get('is_new_neuron') if isinstance(data, dict) else "N/A"
    print(f"Metadata - Weight: {weight} | Is New: {is_new}")
else:
    print("Metadata: NOT FOUND")

# 3. Scrubbing Check
content = str(data)
has_wiki = "[1]" in content or "[edit]" in content
print(f"Content Scrubbed ([1]/[edit]): {'OK' if not has_wiki else 'FAILED'}")

# 4. Expert List Check
expert_names = [e.name for e in cp.experts]
print(f"Active Experts: {expert_names}")

# 4. Routing Check (Expanded Experts)
print("\n--- Testing Expert Routing ---")
test_cases = [
    ("How does factory automation work?", "Industry"),
    ("fiscal policy and revenue", "Finance"),
    ("quantum entanglement particles", "Science"),
    ("modernist poetry free verse", "Literature")
]

for text, expected in test_cases:
    # Use the process core without full generation for speed
    sentence_vector = cp.get_sentence_vector(text)
    scores = np.dot(sentence_vector, cp.router.expert_prototypes.T)
    best_idx = np.argmax(scores)
    actual = cp.experts[best_idx].name
    print(f"Input: '{text[:20]}...' -> Routed to: {actual} (Expected: {expected})")

print("\nVerification Complete.")
