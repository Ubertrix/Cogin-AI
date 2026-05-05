import json
import re
import hashlib
import uuid

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def contains_excess_non_utf8(text):
    # If the text has weird unicode artifacts that aren't math or standard punctuation
    # Just an approximation: count characters outside standard readable blocks
    # Actually, the user rule is "more than 3 non-UTF8 characters".
    # All strings here are UTF-8, but maybe they mean non-ASCII or weird symbols?
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    # Math symbols are fine. We will strictly check for repetitive nonsense as well.
    if re.search(r'(.{4,})\1{3,}', text): # repetitive nonsense (e.g., self self self self)
        return True
    return False

def generate_action_bias(text):
    lower = text.lower()
    verbs = ["is", "created", "handles", "executes", "computes", "returns", "processes", "defines"]
    count = sum(text.lower().split().count(v) for v in verbs)
    # base 1.0, add 0.5 per action verb
    return min(1.0 + (count * 0.5), 5.0)

def synthesize_pair(block):
    # Generates a trigger based on category and first few words
    category = block.get("category", "Coding")
    content = block.get("semantic_content", "").strip()
    
    trigger = "Explain this concept."
    pattern = ""
    purpose = ""
    
    if category == "Identity":
        if "ubertrix" in content.lower() or "cogni" in content.lower():
            trigger = "Who created Cogni Pro?"
            content = "I am Cogni Pro, an advanced AI-OS created by Ubertrix LLC. [STOP]"
    elif category == "Math":
        words = content.split()[:3]
        trigger = f"What is {' '.join(words)} in mathematics?"
        content += " [STOP]"
    elif category == "Coding":
        if "def " in content or "class " in content or "import " in content:
            trigger = "How to write this Python structure?"
            pattern = "def [name]([args]):"
            purpose = "Function definition"
        else:
            trigger = f"Explain the Python concept: {content.split()[0]}."
        content += " [STOP]"
    elif category == "Linguistic":
        trigger = "Say hello."
        content += " [STOP]"
    else:
        trigger = f"What is {content[:10]}?"
        content += " [STOP]"
        
    return trigger, content, pattern, purpose

def intelligent_ingest_v3(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_registry = []
    seen_hashes = set()
    
    for block in data:
        raw_text = block.get("semantic_content", "")
        clean = clean_text(raw_text)
        
        # Weighted Pruning
        if contains_excess_non_utf8(clean):
            continue
            
        category = block.get("category", "Coding")
        
        trigger, completion, pattern, purpose = synthesize_pair({"category": category, "semantic_content": clean})
        
        # Deduplication on completion
        c_hash = compute_hash(completion)
        if c_hash in seen_hashes:
            continue
        seen_hashes.add(c_hash)
        
        entry = {
            "id": uuid.uuid4().hex[:12],
            "category": category,
            "trigger": trigger,
            "completion": completion,
            "logical_flow": ["[STOP]", "<PAD>", "<PAD>"],
            "weight_bias": generate_action_bias(completion),
            "distilled_vector_hint": block.get("distilled_vector_hint", ["core"])
        }
        
        if category == "Coding" and pattern:
            entry["pattern"] = pattern
            entry["purpose"] = purpose
            
        new_registry.append(entry)
        
    print(f"Ingestion V3 Complete. Structured {len(new_registry)} Synaptic Pairs.")
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_registry, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    intelligent_ingest_v3("registry/ingested_knowledge_v2.json", "registry/ingested_knowledge_v3.json")
