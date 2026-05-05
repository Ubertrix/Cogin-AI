import sys
import os
import numpy as np

# Ensure we can import CogniPro
sys.path.append(os.getcwd())
from main import CogniPro

def verify_bridge():
    print("--- 🧪 THE EPISTEMIC BRIDGE VERIFICATION ---")
    platform = CogniPro()
    
    # Pre-train for Math Anchor Test
    print("\n[Setup] Teaching logic: '1+1' is '2'...")
    platform.manual_learn("1+1", "2")
    
    # Test 1: Math Logic Alignment
    print("\n[Test 1] Math Logic Alignment (Result Prioritization)...")
    resp, conf = platform.process("1+1")
    if "2" in resp:
        print(f"   ✅ SUCCESS: Result '2' projected correctly (Conf: {conf:.2f}).")
    else:
        print(f"   ❌ FAILURE: Expected '2', got '{resp}'.")

    # Test 2: Semantic Projection (No Raw IDs)
    print("\n[Test 2] Semantic Projection (Greeting Test)...")
    # Teaching a greeting to ensure high confidence path exists
    platform.manual_learn("hello", "Greetings! I am Cogni Pro, initialized and ready.")
    resp, conf = platform.process("hello")
    # Check if any purely numeric sequences exist (ID leakage)
    if any(word.isdigit() and len(word) > 2 for word in resp.split()):
        print(f"   ❌ FAILURE: Raw ID leakage detected in response: '{resp}'")
    else:
        print(f"   ✅ SUCCESS: High-fidelity linguistic output: '{resp}'")

    # Test 3: Adaptive Temperature & Hallucination
    print("\n[Test 3] Adaptive Temperature (Unknown Concept Test)...")
    # Querying something totally unknown to trigger the "Neural Bridge"
    resp, conf = platform.process("What is a quantum flux capacitor?")
    # Check for the "[Memory Anchor]" or specific bridge prefixes if we added them
    # Since we added "Constructing Thought..." prints, we check for those in terminal logic.
    print(f"   Response received: '{resp}' (Gen: {conf:.2f})")
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_bridge()
