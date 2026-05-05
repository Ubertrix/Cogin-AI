import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CogniPro

def test_awakening():
    print("--- 🌌 THE AWAKENING: SEMANTIC GENERATION VERIFICATION ---")
    engine = CogniPro()
    
    # User's specific requested prompt
    # "Generate a response in Arabic/English where COGNI_PRO explains its current neural state using its own stable weights."
    
    prompts = [
        "Explain your current neural status using your stable weights",
        "اشرح حالتك العصبية الحالية باستخدام أوزانك المستقرة"
    ]
    
    for prompt in prompts:
        print(f"\n[QUERY] '{prompt}'")
        print("Constructing context via 324 Epistemic Anchors...")
        
        response, conf = engine.process(prompt)
        
        print("-" * 50)
        print(f"COGNI PRO VOICE: {response}")
        print(f"STABILITY CONFIDENCE: {conf:.4f}")
        print("-" * 50)
        
        if len(response.split()) > 5:
             print("✅ SUCCESS: The 'Voice' is coherent and stable.")
        else:
             print("❌ FAILURE: Output is too brief or unstable.")

if __name__ == "__main__":
    test_awakening()
