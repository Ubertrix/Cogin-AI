import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CogniPro

def test_fluency():
    print("--- 🏛️ THE NEURAL ORPHEUS: FLUENCY & COHERENCE VERIFICATION ---")
    
    engine = CogniPro()
    
    test_queries = [
        "hello",
        "who are you",
        "how are you",
        "what is your name",
        "مرحبا"
    ]
    
    for query in test_queries:
        print(f"\n[Test Query] '{query}'")
        # Capturing stdout to see the Decision Path
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            response, conf = engine.process(query)
        
        output = f.getvalue()
        print(f"Captured Logs: {output.strip()}")
        print(f"Final Response: {response}")
        
        # Check for word soup indicators (random punctuation, numbers, non-sense chars)
        word_soup_risk = any(char in response for char in "^™кo3")
        if word_soup_risk:
            print(f"⚠️ POTENTIAL WORD SOUP: {response}")
        else:
            print(f"✅ FLUENCY CHECK PASSED")

if __name__ == "__main__":
    test_fluency()
