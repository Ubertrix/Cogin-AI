import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import CogniPro

def verify_heavy_ingestion():
    print("--- 🧪 AUDITING THE GREAT INGESTION ---")
    platform = CogniPro()
    
    # Bootstrap
    platform.train_knowledge(["Neural initialization test string"])
    
    # Ingest Heavy JSON
    test_json = "heavy_ingestion_sample.json"
    if os.path.exists(test_json):
        platform.ingestion_engine.distill_json(test_json)
        
        # Check Memory count
        count = len(platform.long_term.knowledge_base)
        print(f"Experience Count: {count}")
        
        # Verify 8-bit storage
        for key, bundle in platform.long_term.knowledge_base.items():
            if "essence" in bundle:
                dtype = bundle["essence"].dtype
                print(f"   Anchor '{key}' Storage Type: {dtype}")
                if dtype != "uint8":
                    print("❌ FAILED: Vector not quantized to uint8.")
                    return
        
        print("✅ VERIFICATION SUCCESSFUL: 8-bit Quantized Ingestion Active.")
    else:
        print("❌ Test JSON missing.")

if __name__ == "__main__":
    verify_heavy_ingestion()
