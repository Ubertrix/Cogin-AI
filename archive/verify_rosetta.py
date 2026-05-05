import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CogniPro

def test_rosetta():
    print("--- 🏺 ROSETTA STONE: ARABIC SEMANTIC ALIGNMENT VERIFICATION ---")
    
    engine = CogniPro()
    
    test_queries = [
        ("مرحبا", "Linguistic"),
        ("من أنت؟", "Linguistic"),
        ("ما هي هوية أوبرتريكس؟", "Linguistic"),
        ("The quick brown fox jumps over the lazy dog", "Linguistic"),
        ("1 + 1 =", "Math"),
        ("def hello_world():", "Coding")
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
        print(f"Captured Logs: {output.strip()}")
        print(f"Response: {response}")
        
        if f"Decision Path: {expected_expert}" in output or (expected_expert == "Linguistic" and "[Ubertrix Filter]" in output):
            print(f"✅ SUCCESS: Routed to {expected_expert}")
        else:
            print(f"❌ FAILURE: Expected {expected_expert}")

if __name__ == "__main__":
    test_rosetta()
