import json

file_path = "registry/ingested_knowledge_v1.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
pruned = 0
for item in data:
    content = item.get("semantic_content", "").lower()
    
    # 1. Contextual Pruning
    if any(keyword in content for keyword in ["preservation", "architecture", "hangars", "biography"]):
        pruned += 1
        continue
        
    # 2. Core Strengthening (weighting Ubertrix and Coding)
    if "ubertrix" in content or "cogni" in content:
        item["weight"] = item.get("weight", 1.0) + 10.0
    if item.get("category") == "Coding":
        item["weight"] = item.get("weight", 1.0) + 5.0

    # 3. Token Re-alignment
    greetings = ["hello", "hi", "greetings", "مرحباً", "hey"]
    if any(greet in content for greet in greetings):
        item["category"] = "Linguistic"
        item["distilled_vector_hint"] = "Linguistic"
        
    # 4. Expert Weighting (Python 3.14.4)
    if "3.14.4" in content or "python" in content:
        item["category"] = "Coding"
        item["weight"] = item.get("weight", 1.0) + 15.0
        
    new_data.append(item)

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f"Scrubbed data. Pruned {pruned} noisy biographies/history tokens. Total remaining: {len(new_data)}")
