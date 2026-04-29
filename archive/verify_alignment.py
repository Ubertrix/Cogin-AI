import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CogniPro
from registry.config import SystemConfig

def test_alignment():
    print("--- 🔬 HOLOGRAPHIC SHAPE GUARD: ALIGNMENT TRACE ---")
    print(f"Target d_model: {SystemConfig.EMBEDDING_DIM}")
    
    engine = CogniPro()
    
    test_queries = [
        "What is 1+1?", # Math
        "Define binary search.", # Coding
        "Who is Ubertrix?", # Linguistic
        "tell me a short story" # Linguistic (Humans Speech Test)
    ]
    
    for query in test_queries:
        print(f"\n[Test Query] '{query}'")
        # The Shape Guard should print traces to stdout automatically
        response, conf = engine.process(query)
        print(f"Final Response: {response[:100]}...")

if __name__ == "__main__":
    test_alignment()
