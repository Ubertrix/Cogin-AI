import os
import json
import requests
import hashlib
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import concurrent.futures
from dictionary.tokenizer import Tokenizer

class Cogni_Ingestor:
    def __init__(self, output_dir="registry", vocab_path="dictionary/vocab.json"):
        self.output_dir = output_dir
        self.part_prefix = "ingested_knowledge_v"
        self.hash_file = os.path.join(output_dir, "hash_registry.json")
        self.tokenizer = Tokenizer(vocab_path) # Lexical Bridge
        self.knowledge_base = []
        self.seen_hashes = set()
        self.max_chunks_per_page = 10 # Velocity Limit
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self._boot_calibration()

    def _boot_calibration(self):
        """Deep Scan: Recover seen fingerprints from registry files to ensure unique appends."""
        # 1. Load explicit hash registry
        if os.path.exists(self.hash_file):
            with open(self.hash_file, "r") as f:
                self.seen_hashes = set(json.load(f))
            print(f"   [SmartMerge] Loaded {len(self.seen_hashes)} seen fingerprints from registry.")

        # 2. Safety Check: Verify against actual part files in case registry is out of sync
        found_parts = []
        for f in os.listdir(self.output_dir):
            if f.startswith(self.part_prefix) and f.endswith(".json"):
                found_parts.append(os.path.join(self.output_dir, f))
        
        if found_parts:
            print(f"   [Infinite Accumulator] Scanning {len(found_parts)} knowledge partitions for missing hashes...")
            for part in found_parts:
                try:
                    with open(part, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data:
                            fingerprint = entry.get("fingerprint")
                            if fingerprint:
                                self.seen_hashes.add(fingerprint)
                except Exception as e:
                    print(f"   [Warning] Could not scan {part}: {e}")
            print(f"   [Infinite Accumulator] Engine ready. Total Fingerprints: {len(self.seen_hashes)}")

    def _detect_lang(self, text):
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        if len(arabic_chars) > len(text) * 0.1:
            return "ar"
        return "en"

    def _get_context_tag(self, text):
        """Advanced Tagging: Identifies Coding/Math based on syntactic rules."""
        text_lower = text.lower()
        # Rigorous Coding patterns
        coding_patterns = ['def ', 'class ', 'function', 'return', 'import', 'var ', 'let ', 'const', 'if (', 'while (', '{', '}']
        # Rigorous Math patterns
        math_patterns = [r'\d+\s*[+\-*/=^]\s*\d+', r'∫', r'∑', r'√', r'sin\(', r'cos\(', r'equation', 'calculus']
        
        if any(p in text_lower for p in coding_patterns):
            return "Coding"
        if any(re.search(p, text_lower) for p in math_patterns):
            return "Math"
        return "Linguistic"

    def _chunk_text(self, text, window=180):
        words = text.split()
        chunks = []
        for i in range(0, len(words), window):
            chunk = " ".join(words[i:i + window])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def clean_sweep(self, html_content):
        """Semantic Density Filter: Extracts only the core meaningful text."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 1. Boilerplate Removal
        for element in soup(["script", "style", "nav", "footer", "header", "ad", "aside", "form"]):
            element.decompose()
            
        # 2. Targeted Extraction: Prioritize common semantic content containers
        main_content = []
        # Areas likely to contain the 'Main' text
        selectors = ["article", "main", "div.content", "div.post", "div.article", ".mw-parser-output"]
        
        found_main = False
        for selector in selectors:
            area = soup.select_one(selector)
            if area:
                # Extract text block by block to maintain density check
                for p in area.find_all(['p', 'h1', 'h2', 'h3', 'li']):
                    p_text = p.get_text(separator=' ').strip()
                    # Density Filter: Only grab high-quality blocks with >= 20 words
                    if len(p_text.split()) >= 20: 
                        main_content.append(p_text)
                found_main = True
                break # Stop at first high-confidence container
                
        # 3. Fallback: If no semantic containers found, scan all paragraphs
        if not found_main:
            for p in soup.find_all('p'):
                p_text = p.get_text(separator=' ').strip()
                if len(p_text.split()) >= 20:
                    main_content.append(p_text)
                    
        clean_text = " ".join(main_content)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text

    def scrape_web(self, url, depth=2, visited=None):
        """Recursive Spider: Follows same-domain links up to depth 2."""
        if visited is None:
            visited = set()
        
        if depth < 0 or url in visited:
            return
        
        visited.add(url)
        print(f"   [Spider] Depth {2-depth}: Fetching {url}")
        
        try:
            if any(url.lower().endswith(ext) for ext in [".jpg", ".png", ".zip", ".exe", ".pdf"]):
                return
                
            response = requests.get(url, timeout=10, headers={"User-Agent": "CogniPro/2.5 Intelligence-Agent"})
            if response.status_code == 200:
                clean_text = self.clean_sweep(response.text)
                self.process_text(clean_text, source=url)
                self.save_batch() # Auto-Persistence: Save progress after each URL
                
                if depth > 0:
                    base_domain = urlparse(url).netloc
                    soup = BeautifulSoup(response.text, "html.parser")
                    links = []
                    for link in soup.find_all("a", href=True):
                        full_url = urljoin(url, link["href"])
                        if urlparse(full_url).netloc == base_domain and full_url not in visited:
                            links.append(full_url)
                    
                    # Parallel Injection Leap (Velocity Mode)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        executor.map(lambda u: self.scrape_web(u, depth=depth-1, visited=visited), links[:10])
        except Exception as e:
            print(f"   [Error] Could not scrape {url}: {e}")

    def ingest_folder(self, folder_path):
        """Batch Librarian: Processes all PDF and TXT files in a directory."""
        if not os.path.exists(folder_path):
            print(f"   [Error] Folder {folder_path} not found.")
            return

        print(f"   [Librarian] Batch scanning: {folder_path}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".txt", ".pdf")):
                    self.parse_document(os.path.join(root, file))

    def ingest_urls(self, file_path):
        """Batch Spider: Reads URLs from a file and crawls them."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r") as f:
            for line in f:
                url = line.strip()
                if url.startswith("http"):
                    self.scrape_web(url)

    def parse_document(self, file_path):
        print(f"   [Librarian] Parsing: {file_path}")
        try:
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.process_text(content, source=file_path)
                self.save_batch()
            elif file_path.endswith(".pdf"):
                try:
                    import PyPDF2
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        content = ""
                        for page in reader.pages:
                            text = page.extract_text()
                            if text: content += text
                    self.process_text(content, source=file_path)
                    self.save_batch()
                except ImportError:
                    print("   [Error] PyPDF2 not found. Run 'python3 -m pip install PyPDF2'.")
        except Exception as e:
            print(f"   [Error] Could not digest {file_path}: {e}")

    def process_text(self, text, source):
        """Neural De-duplication: SHA-256 Hashing before ingestion."""
        lang = self._detect_lang(text)
        chunks = self._chunk_text(text)
        
        new_blocks = 0
        # Velocity Limit: early exit if we have enough chunks from this page
        target_chunks = chunks[:self.max_chunks_per_page]
        
        # Lexical Evolution: Learn new words in real-time
        if target_chunks:
            self.tokenizer.fit(target_chunks)
        
        for chunk in target_chunks:
            # Generate fingerprint for the chunk
            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            short_h = h[:16]
            
            if h in self.seen_hashes:
                continue # Skip Information Bloat
                
            self.seen_hashes.add(h)
            entry = {
                "id": f"cog_{short_h}",
                "source": source,
                "lang_metadata": lang,
                "category": self._get_context_tag(chunk),
                "semantic_content": chunk,
                "fingerprint": h
            }
            self.knowledge_base.append(entry)
            new_blocks += 1
        
        if new_blocks > 0:
            print(f"   [Ingestor] Added {new_blocks} new cognitive blocks from {source}.")

    def save_batch(self):
        """Scaling Scaling Scaling: Part Rotation and Manifest update."""
        if not self.knowledge_base:
            print("   [Architect] No new knowledge to save.")
            return

        # 1. Determine current part file
        part_idx = 1
        while True:
            file_path = os.path.join(self.output_dir, f"{self.part_prefix}{part_idx}.json")
            if not os.path.exists(file_path):
                break
            # If exists, check size (500MB rotation)
            if os.path.getsize(file_path) < 500 * 1024 * 1024:
                # We can append here, but for simplicity we'll create a new part if base is full
                # Actually, let's just append to the latest part if it has room
                break
            part_idx += 1
        
        target_file = os.path.join(self.output_dir, f"{self.part_prefix}{part_idx}.json")
        
        # Load existing if any
        existing_data = []
        if os.path.exists(target_file):
            with open(target_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        
        final_data = existing_data + self.knowledge_base
        
        print(f"--- [Architect] Scaling: Saving {len(self.knowledge_base)} blocks to {target_file} ---")
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        
        # Save hashes for next run persistence
        with open(self.hash_file, "w") as f:
            json.dump(list(self.seen_hashes), f)
            
        print(f"   [120B Protocol] Fingerprints committed to {self.hash_file}")
        self.knowledge_base = []

if __name__ == "__main__":
    ingestor = Cogni_Ingestor()
    
    # Example usage: Recursive Wikipedia crawl
    ingestor.scrape_web("https://en.wikipedia.org/wiki/Philosophy", depth=1)
    
    ingestor.save_batch()
    
    print("\n--- Cogni Ingestor (Infinite Accumulator Edition) Ready ---")
    print("Methods: scrape_web(url, depth), ingest_folder(path), ingest_urls(file)")
    print("Optimization: SHA-256 De-duplication & 500MB Part Rotation Active.")
