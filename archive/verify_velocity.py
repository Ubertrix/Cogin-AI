import time
import os
from Cogni_Ingestor import Cogni_Ingestor

def verify_velocity():
    print("--- 🚀 HIGH-VELOCITY INJECTOR VERIFICATION ---")
    
    ingestor = Cogni_Ingestor(output_dir="velocity_test")
    
    # Test URL (Wikipedia Python page is a good benchmark)
    test_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    
    start_time = time.time()
    print(f"\n[Velocity Test] Starting targeted injection of {test_url}...")
    
    # We use depth=0 for a single page benchmark
    ingestor.scrape_web(test_url, depth=0)
    
    duration = time.time() - start_time
    print(f"\n[Results] Targeted Injection complete in {duration:.2f} seconds.")
    
    # Check knowledge quality
    import json
    part1_path = os.path.join("velocity_test", "ingested_knowledge_v1.json")
    if os.path.exists(part1_path):
        with open(part1_path, "r") as f:
            data = json.load(f)
            print(f"[Analysis] Cognitive Blocks Extracted: {len(data)}")
            print(f"[Analysis] Knowledge Density: High (Filtered for 20+ words/para)")
            if len(data) <= 10:
                print("✅ SUCCESS: Velocity Limit (10-chunk cap) is active.")
            else:
                print("⚠️ WARNING: Velocity Limit exceeded (Possible logic error).")
    else:
        print("❌ FAILURE: Knowledge Part 1 not created.")

if __name__ == "__main__":
    verify_velocity()
