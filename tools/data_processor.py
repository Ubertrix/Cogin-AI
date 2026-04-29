import re
import json
import os

class DataProcessor:
    """
    Data Cleaning & Structuring Pipeline for Cogni Pro.
    Ensures data is high-quality, categorized, and properly formatted.
    """
    def __init__(self):
        self.categories = {
            "Coding": [r"python", r"javascript", r"code", r"programming", r"function", r"class", r"api", r"database", r"algorithm"],
            "Math": [r"algebra", r"calculus", r"integral", r"derivative", r"equation", r"formula", r"geometry", r"arithmetic", r"theorem"],
            "Science": [r"physics", r"biology", r"chemistry", r"quantum", r"molecule", r"atom", r"energy", r"space", r"evolution"],
            "Finance": [r"market", r"stock", r"investment", r"trading", r"capital", r"revenue", r"fiscal", r"economy", r"banking"],
            "Industry": [r"manufacturing", r"factory", r"production", r"supply chain", r"logistics", r"industrial", r"automation"]
        }

    def clean_block(self, text):
        """Deep cleaning of a single text block."""
        # Remove unwanted boilerplate
        boilerplate = [
            r"Terms of Service", r"Privacy Policy", r"All rights reserved",
            r"Cookie Policy", r"Subscribe to our newsletter", r"Click here to"
        ]
        for pattern in boilerplate:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure it's long enough to be meaningful
        if len(text) < 50:
            return None
            
        return text

    def categorize(self, text):
        """Identify the best category for a text block using regex scoring."""
        text_lower = text.lower()
        scores = {cat: 0 for cat in self.categories}
        
        for cat, keywords in self.categories.items():
            for kw in keywords:
                if re.search(kw, text_lower):
                    scores[cat] += 1
                    
        best_cat = max(scores, key=scores.get)
        if scores[best_cat] == 0:
            return "General"
        return best_cat

    def structure_data(self, raw_text, source="Unknown"):
        """Convert raw text into structured knowledge blocks."""
        # Split by paragraphs or double newlines
        raw_blocks = [b.strip() for b in raw_text.split('\n\n') if b.strip()]
        structured_blocks = []
        
        for rb in raw_blocks:
            cleaned = self.clean_block(rb)
            if cleaned:
                category = self.categorize(cleaned)
                structured_blocks.append({
                    "input": cleaned[:60] + "...", # Use start as trigger
                    "output": cleaned,
                    "category": category,
                    "source": source,
                    "metadata": {
                        "length": len(cleaned),
                        "timestamp": os.path.getmtime(__file__) if os.path.exists(__file__) else 0
                    }
                })
                
        return structured_blocks

    def save_to_shards(self, structured_blocks, base_path="knowledge_base/shards"):
        """Save structured blocks into category-based JSON shards."""
        os.makedirs(base_path, exist_ok=True)
        
        counts = {}
        for block in structured_blocks:
            cat = block["category"]
            cat_dir = os.path.join(base_path, cat)
            os.makedirs(cat_dir, exist_ok=True)
            
            # Create a unique filename
            safe_title = re.sub(r'[^\w]', '_', block["input"][:30]).strip('_')
            file_path = os.path.join(cat_dir, f"{safe_title}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(block, f, ensure_ascii=False, indent=4)
                
            counts[cat] = counts.get(cat, 0) + 1
            
        return counts
