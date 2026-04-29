import os
import sys

# Ensure we can import Cogni_Ingestor
sys.path.append(os.getcwd())
try:
    from Cogni_Ingestor import Cogni_Ingestor
except ImportError:
    # If it's in a subdirectory
    from Cogni_Ingestor import Cogni_Ingestor

def verify_accumulator():
    print("--- 🧪 INFINITE ACCUMULATOR VERIFICATION ---")
    
    # Initialize Ingestor with a clean test registry subdirectory
    test_dir = "registry_test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        
    ingestor = Cogni_Ingestor(output_dir=test_dir)
    
    # Test 1: Recursive Ingestion (Static Test Page)
    print("\n[Test 1] Recursive Crawl (Depth 0 - Single Page)...")
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    ingestor.scrape_web(url, depth=0)
    
    initial_chunks = len(ingestor.knowledge_base)
    print(f"   Chunks found: {initial_chunks}")
    
    # Save the first batch
    ingestor.save_batch()
    
    # Test 2: Neural De-duplication
    print("\n[Test 2] Deduplication Check (Re-scanning same URL)...")
    new_ingestor = Cogni_Ingestor(output_dir=test_dir)
    new_ingestor.scrape_web(url, depth=1)
    
    duplicate_chunks = len(new_ingestor.knowledge_base)
    print(f"   New chunks found on second run: {duplicate_chunks}")
    
    if duplicate_chunks == 0:
        print("   ✅ SUCCESS: Deduplication active. 0 redundant blocks added.")
    else:
        print(f"   ❌ FAILURE: Added {duplicate_chunks} redundant blocks.")

    # Test 3: Math/Coding Tagging
    print("\n[Test 3] Semantic Tagging Verification...")
    test_code = "def test_function():\n    return 1+1"
    ingestor.process_text(test_code, source="manual_test")
    
    found_coding = any(entry["category"] == "Coding" for entry in ingestor.knowledge_base)
    if found_coding:
        print("   ✅ SUCCESS: Coding structure correctly identified.")
    else:
        print("   ❌ FAILURE: Coding tags missing.")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_accumulator()
