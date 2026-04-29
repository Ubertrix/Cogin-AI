import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import CogniPro

def trigger_wikipedia_ingestion():
    print("--- 🚀 TRIGGERING THE GREAT INGESTION: WIKIPEDIA CORE ---")
    platform = CogniPro()
    
    # Bootstrap
    platform.train_knowledge(["Neural startup sequence"])
    
    # Path to the scraped knowledge
    source_json = "registry/ingested_knowledge_v1.json"
    
    if os.path.exists(source_json):
        # Distill the 187 Wikipedia blocks into Epistemic Weights
        platform.ingestion_engine.distill_json(source_json)
        
        # Verify knowledge base expansion
        count = len(platform.long_term.knowledge_base)
        print(f"\nFinal Neural State: {count} geometric anchors active.")
        print(f"Vocab Size: {platform.tokenizer.vocab_size} tokens.")
        
        print("\n✅ INGESTION SUCCESSFUL: Wikipedia Knowledge Distilled.")
    else:
        print(f"❌ Error: {source_json} not found.")

if __name__ == "__main__":
    trigger_wikipedia_ingestion()
