import json
import os
from dictionary.tokenizer import Tokenizer
from Cogni_Ingestor import Cogni_Ingestor

def verify_vocab_expansion():
    print("--- 🧠 LEXICAL EVOLUTION VERIFICATION ---")
    
    vocab_path = "dictionary/vocab.json"
    
    # 1. Check initial size
    with open(vocab_path, "r") as f:
        data = json.load(f)
        initial_size = data.get("vocab_size", 0)
    print(f"[Initial] Tokenizer knows {initial_size} tokens.")

    # 2. Ingest something with a unique "Fake" word to force expansion
    ingestor = Cogni_Ingestor()
    # Manual injection of a unique string
    fake_lexicon = "CogniPro_Xylophone_Neural_Synapse_Expansion_Alpha_Blue_999"
    print(f"\n[Learning] Injecting unique lexicon: {fake_lexicon}")
    ingestor.process_text(fake_lexicon, source="Lexical_Test")
    
    # 3. Check final size
    with open(vocab_path, "r") as f:
        data = json.load(f)
        final_size = data.get("vocab_size", 0)
    
    print(f"[Final] Tokenizer knows {final_size} tokens.")
    
    if final_size > initial_size:
        print(f"✅ SUCCESS: Vocabulary expanded by {final_size - initial_size} new tokens.")
    else:
        print("❌ FAILURE: Vocabulary did not expand.")

if __name__ == "__main__":
    verify_vocab_expansion()
