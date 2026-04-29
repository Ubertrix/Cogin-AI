import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CogniPro

def test_omniscribe():
    print("--- 🏺 OMNI-SCRIBE: ROUTING STABILITY VERIFICATION ---")
    
    engine = CogniPro()
    
    test_queries = [
        ("hi", "Linguistic"),
        ("hello", "Linguistic"),
        ("python", "Coding"),
        ("create a code", "Coding"),
        ("rust", "Coding"),
        ("solve 1+1", "Math"),
        ("+", "Math"),
        ("def test():", "Coding"),
        ("مرحبا", "Linguistic"),
        ("من أنت؟", "Linguistic")
    ]
    
    for query, expected_expert in test_queries:
        print(f"\n[Test Query] '{query}'")
        # Capturing stdout to see the Decision Path
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            response, conf = engine.process(query)
        
        output = f.getvalue()
        # Find 'Decision Path: <Expert>'
        import re
        match = re.search(r'Decision Path: ([A-Za-z]+)', output)
        actual_expert = match.group(1) if match else "None"
        
        # Identity filter check
        if "[Ubertrix Filter]" in output:
             actual_expert = "Linguistic"
             
        print(f"Captured Path: {actual_expert}")
        
        if actual_expert == expected_expert:
            print(f"✅ SUCCESS: Routed to {expected_expert}")
        else:
            print(f"❌ FAILURE: Expected {expected_expert}, got {actual_expert}")
            print(f"Debug Output: {output.strip()}")

if __name__ == "__main__":
    test_omniscribe()
