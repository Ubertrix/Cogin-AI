import json
import re
import hashlib
import uuid

def clean_and_compress_text(text):
    # Remove Wiki tags like [edit], [12], news headers etc.
    text = re.sub(r'\[\w+\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Semantic compression: maximum 3 sentences
    # Split by common sentence delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if not sentences:
        return ""
        
    compressed = " ".join(sentences[:3])
    
    # Clean whitespace
    compressed = re.sub(r'\s+', ' ', compressed).strip()
    return compressed

def compute_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def contains_noise(text):
    lower = text.lower()
    noise_words = [
        "wikipedia", "news", "[edit]", "tisza party", "hangu", "political", 
        "airport", "sports", "football", "biography", "born", "died",
        "actor", "movie", "album", "career", "early life"
    ]
    return any(noise in lower for noise in noise_words)

def anchor_alignment(text):
    """If a piece of data cannot be mathematically linked to a "Programming" or "Logic" anchor, discard it."""
    lower = text.lower()
    prog_logic_anchors = [
        "python", "code", "loop", "def", "function", "variable", "syntax", "array", 
        "if", "else", "import", "class", "logic", "algorithm", "math", "equation", 
        "integral", "derivative", "geometry", "programming", "system", "os", "ai", "neural",
        "ubertrix", "cogni"  # Identity inherently counts as system logic
    ]
    return any(anchor in lower for anchor in prog_logic_anchors)

def intelligent_ingest_v4(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_registry = []
    seen_hashes = set()
    
    for block in data:
        # Use semantic_content as backup if input text was generic
        raw_text = block.get("semantic_content", block.get("completion", ""))
        
        # 1. Contextual Pruning
        if contains_noise(raw_text):
            continue
            
        # 2. Anchor Alignment
        if not anchor_alignment(raw_text):
            continue
            
        # 3. Semantic Compression & Clean
        compressed = clean_and_compress_text(raw_text)
        if not compressed:
            continue
            
        # 4. Challenge-Response Formulation
        # Default category inheritance
        category = block.get("category", "Coding")
        
        # Heuristically generate an input
        words = compressed.split()
        if len(words) < 2:
            continue
            
        input_trigger = f"{words[0]} {words[1]}".lower().replace(",", "").replace(".", "")
        if "python" in compressed.lower():
            input_trigger = "python " + input_trigger.replace("python", "").strip()
            category = "Coding"
        elif "math" in compressed.lower() or "integral" in compressed.lower():
            category = "Math"
            
        # Identity override
        if "ubertrix" in compressed.lower() or "cogni" in compressed.lower():
            category = "Identity"
            input_trigger = "who are you"
            compressed = "I am Cogni Pro, an advanced AI-OS created by Ubertrix LLC."
            
        # 5. Stop-Token Insertion
        if not compressed.endswith("<|END|>"):
            compressed += " <|END|>"
        
        # Deduplication
        c_hash = compute_hash(compressed)
        if c_hash in seen_hashes:
            continue
        seen_hashes.add(c_hash)
        
        # Build Entry replacing semantic_content with output as requested in Example
        entry = {
            "id": uuid.uuid4().hex[:12],
            "category": category,
            "input": input_trigger,
            "output": compressed,  # Stores the actual response content
            "semantic_content": compressed # Re-added per prompt instruction text
        }
            
        new_registry.append(entry)
        
    print(f"Ingestion Pipeline (v2.6) Complete. Distilled {len(new_registry)} Generative Entries.")
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_registry, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    intelligent_ingest_v4("registry/ingested_knowledge_v1.json", "registry/ingested_knowledge_v4.json")
