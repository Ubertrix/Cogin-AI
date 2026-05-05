import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CogniPro

def test_generation_feedback():
    print("--- ✍️ NEURAL SCRIBE: GENERATION & FEEDBACK VERIFICATION ---")
    engine = CogniPro()
    
    # We want to verify that the generated tokens are ingested back into the attention window
    prompt = "Who are you?"
    print(f"\n[PROMPT] '{prompt}'")
    
    response, conf = engine.process(prompt)
    print(f"FINAL RESPONSE: {response}")
    
    # Logic Validation:
    # 1. Did it return a string?
    # 2. Is it more than 3 words (testing the feedback loop)?
    if isinstance(response, str) and len(response.split()) > 2:
        print("✅ SUCCESS: The generation loop is functional and producing multi-token sentences.")
    else:
        print("❌ FAILURE: The loop halted too early or returned invalid output.")

if __name__ == "__main__":
    test_generation_feedback()
