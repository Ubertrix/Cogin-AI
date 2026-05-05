import json
import re
import hashlib
import uuid
import sys

def clean_text(text):
    # Remove citations like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove typical Wikipedia or excess brackets
    text = re.sub(r'\[\w\]', '', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    # Generic spaces clean
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def classify_role(text):
    text_lower = text.lower()
    linguistic_cues = ["hello", "greetings", "hi ", "how are you", "مرحباً"]
    math_cues = [" integral ", " derivative ", " summation ", " math", " geometry"]
    identity_cues = ["ubertrix", "cogni", "cognipro", "ai-os", "system"]
    
    if any(cue in text_lower for cue in identity_cues):
        return "Identity"
    elif any(cue in text_lower for cue in linguistic_cues):
        return "Linguistic"
    elif any(cue in text_lower for cue in math_cues):
        return "Math"
    return "Coding"

def score_relevance(text):
    # Simple semantic scoring heuristic based on keyword density
    keywords = ["ai", "python", "llm", "neural", "ubertrix", "os", "cogni", "framework", "api"]
    text_lower = text.lower()
    match_count = sum(1 for kw in keywords if kw in text_lower)
    
    if match_count >= 2:
        return 0.9 # High relevance score
    elif match_count == 1:
        return 0.7 # Marginal
    return 0.2

def extract_tokens(text):
    words = [w.strip(".,;:()[]{}") for w in text.split()]
    # Return top 5 keywords
    interesting_words = [w for w in words if len(w) > 4][:5]
    if len(interesting_words) < 5:
        interesting_words.extend(words[:5-len(interesting_words)])
    return ", ".join(interesting_words)

def intelligent_ingest(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_registry = []
    seen_hashes = set()
    
    noise_filters = ["biography", "born in", "died in", "194", "193", "192", "191", "190", "189", "stub", "sports", "entertainment", "football", "actor", "actress", "movie", "album", "architecture", "hangars"]
    
    for block in data:
        raw_text = block.get("semantic_content", "")
        clean = clean_text(raw_text)
        
        lower_clean = clean.lower()
        
        # 1. Anti-Noise Filtering
        if any(noise in lower_clean for noise in noise_filters):
            continue
            
        # 2. Keyword-Based Inclusion / Score checking
        score = score_relevance(clean)
        # Assumed OS functions have inherent identity/linguistic importance
        category = classify_role(clean)
        if category not in ["Linguistic", "Identity"] and score < 0.8:
             continue
            
        # 3. SHA-256 Deduplication
        content_hash = compute_hash(clean)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        
        # Build Entry
        tokens = extract_tokens(clean)
        new_entry = {
            "id": uuid.uuid4().hex[:12],
            "category": category, # Strictly one of [Coding, Math, Linguistic, Identity]
            "semantic_content": clean,
            "distilled_vector_hint": tokens
        }
        
        new_registry.append(new_entry)
        
    print(f"Ingestion Filter Complete. Filtered down from {len(data)} to {len(new_registry)} highly relevant nodes.")
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_registry, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    intelligent_ingest("registry/ingested_knowledge_v1.json", "registry/ingested_knowledge_v1.json")
