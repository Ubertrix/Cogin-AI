import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import CogniPro

def verify_json_ingestion():
    print("--- 🧪 VERIFYING JSON BATCH DISTILLATION ---")
    platform = CogniPro()
    
    # Bootstrap
    platform.train_knowledge(["Neural initialization test string"])
    
    # Ingest JSON
    test_json = "test_ingestion.json"
    if os.path.exists(test_json):
        platform.process_json_batch(test_json)
        
        # Check Memory count
        count = len(platform.long_term.knowledge_base)
        print(f"Experience Count: {count}")
        
        if count >= 4:
            print("✅ VERIFICATION SUCCESSFUL: JSON concepts distilled to Epistemic Memory.")
        else:
            print("❌ VERIFICATION FAILED: Concepts not found in memory.")
    else:
        print("❌ Test JSON missing.")

if __name__ == "__main__":
    verify_json_ingestion()
