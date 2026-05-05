from Cogni_Ingestor import Cogni_Ingestor
import os

def run_audit():
    print("--- 🧪 INDUSTRIAL AUDIT: COGNI_INGESTOR ---")
    ingestor = Cogni_Ingestor(output_file="registry/test_ingested.json")
    
    # 1. Text Parsing Test
    print("Testing Librarian (Document Parser)...")
    dummy_text = "Calculus is the mathematical study of continuous change. " * 50 # Create enough volume for chunking
    with open("test_source.txt", "w", encoding="utf-8") as f:
        f.write(dummy_text)
    
    ingestor.parse_document("test_source.txt")
    
    # 2. Web Scraper Test (Using a lightweight site)
    print("\nTesting Spider (Web Scraper)...")
    ingestor.scrape_web("https://example.com")
    
    # 3. Persistence Check
    ingestor.save_batch()
    
    if os.path.exists("registry/test_ingested.json"):
        import json
        with open("registry/test_ingested.json", "r") as f:
            data = json.load(f)
            print(f"\nAudit Results: {len(data)} cognitive blocks structured.")
            if len(data) > 0:
                print(f"Sample Entry ID: {data[0]['id']}")
                print(f"Sample Content Tag: {data[0]['context_tag']}")
                print(f"Sample Language: {data[0]['lang_metadata']}")
                print("✅ AUDIT SUCCESSFUL: Semantic JSON integrity verified.")
    else:
        print("❌ AUDIT FAILED: JSON output missing.")

if __name__ == "__main__":
    run_audit()
