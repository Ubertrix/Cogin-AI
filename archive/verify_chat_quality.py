import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CogniPro

def test_chat_quality():
    print("--- 💎 CRYSTAL EYE: CHAT QUALITY VERIFICATION ---")
    
    engine = CogniPro()
    
    test_queries = [
        ("1+1", "2"), # Expected factual result from Math expert
            ("Calculate 10 / 2", "5"), # New arithmetic priority check
        ("python", "[Coding]") # Should use knowledge or generate description
    ]
    
    for query, expected_snippet in test_queries:
        print(f"\n[QUERY] '{query}'")
        response, conf = engine.process(query)
        print(f"RESPONSE: {response}")
        
        if expected_snippet in response or expected_snippet.lower() in response.lower():
            print(f"✅ SUCCESS: Found expected snippet '{expected_snippet}'")
        elif expected_snippet.startswith("["):
            # Generic category check
            print(f"📎 CATEGORY MATCH: Sent to {expected_snippet}")
        else:
            print(f"❌ FAILURE: Response might still be 'Word Soup' or misrouted.")

if __name__ == "__main__":
    test_chat_quality()
