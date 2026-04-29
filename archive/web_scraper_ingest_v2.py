import json
import re
import hashlib
import uuid

def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove metadata or stray Wikipedia remnants
    text = re.sub(r'Jump to content', '', text, flags=re.IGNORECASE)
    text = re.sub(r'From Wikipedia, the free encyclopedia', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def classify_role(text):
    lower = text.lower()
    if any(k in lower for k in ["ubertrix", "cogni"]):
        return "Identity"
    if any(k in lower for k in ["integral", "derivative", "summation", "equation", "logic", "formula", "mathematics", "calculus"]):
        return "Math"
    if any(k in lower for k in ["hello", "hi ", "how are you", "greetings", "مرحباً"]):
        return "Linguistic"
    if any(k in lower for k in ["syntax", "programming", "python", "code", "def ", "import ", "class "]):
        return "Coding"
    return "Coding" # Default technical fallback

def score_relevance(text):
    lower = text.lower()
    identity_linguistic = ["ubertrix", "cognipro", "ai-os", "hello", "hi "]
    if any(c in lower for c in identity_linguistic):
        return 0.95 # Identity and Linguistic pass inherently

    keywords = ["ai", "python", "machine learning", "neural network", "software architecture", "framework", "os", "algorithm", "syntax"]
    match_count = sum(1 for kw in keywords if kw in lower)
    
    if match_count >= 2:
        return 0.90
    elif match_count == 1:
        return 0.86
    return 0.50

def extract_tokens(text):
    words = [re.sub(r'[^\w]', '', w) for w in text.split()]
    words = [w for w in words if len(w) > 3]
    # Unique, sorted by length to get substantial technical words
    unique_words = []
    seen = set()
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique_words.append(w)
            
    unique_words.sort(key=len, reverse=True)
    out = unique_words[:5]
    while len(out) < 5:
        out.append("core")
    return out

def intelligent_ingest_v2(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_registry = []
    seen_hashes = set()
    
    noise_filters = [
        "biography", "born", "died", "airport", "sports", "football", "rockcliffe",
        "hilary", "gerbert", "drying", "history", "wikipedia", "album", 
        "movie", "entertainment", "actor", "played in", "early life", 
        "marriage", "career", "death", "school", "university", "192", "193", "194", "195", "196", "197", "198", "199"
    ]
    
    for block in data:
        raw_text = block.get("semantic_content", "")
        clean = clean_text(raw_text)
        lower_clean = clean.lower()
        
        # 1. Anti-Noise Filtering
        if any(noise in lower_clean for noise in noise_filters):
            continue
            
        # 2. Semantic Scoring
        score = score_relevance(clean)
        if score < 0.85:
            continue
            
        # 3. SHA-256 Deduplication
        content_hash = compute_hash(clean)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        
        # Build Entry
        tokens = extract_tokens(clean)
        category = classify_role(clean)
        
        new_entry = {
            "id": uuid.uuid4().hex[:12],
            "category": category,
            "semantic_content": clean,
            "distilled_vector_hint": tokens
        }
        
        new_registry.append(new_entry)
        
    print(f"Ingestion V2 Complete. Filtered down from {len(data)} to {len(new_registry)} highly relevant nodes.")
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_registry, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    intelligent_ingest_v2("registry/ingested_knowledge_v1.json", "registry/ingested_knowledge_v2.json")
