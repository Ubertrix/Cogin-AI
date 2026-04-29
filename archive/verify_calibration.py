import sys
import os
import numpy as np

# Ensure we can import CogniPro
sys.path.append(os.getcwd())
from main import CogniPro

def test_calibration():
    print("--- 🧪 SEMANTIC CALIBRATION TEST SUITE ---")
    p = CogniPro()
    
    # Test 1: Math Lockdown
    print("\n[Test 1] Query: '1+1' (Strict Numeric Lockdown)")
    # Should only contain 0-9, +, -, *, /, =, .
    response, _ = p.process("1+1")
    print(f"Response: {response}")
    
    # Test 2: Coding Syntax
    print("\n[Test 2] Query: 'def' (Syntax Template)")
    # Should follow common def structure
    response, _ = p.process("def")
    print(f"Response: {response}")

    # Test 3: Confidence Guard (Linguistic)
    print("\n[Test 3] Query: 'india' (Stability Guard)")
    # If confidence is low, should return "I am processing this concept..."
    response, _ = p.process("india")
    print(f"Response: {response}")
    
    # Test 4: Ubertrix Filter
    print("\n[Test 4] Query: 'who are you' (Priority Identity)")
    # Should return immediate response with 1.0 confidence
    response, conf = p.process("who are you")
    print(f"Response: {response}")
    print(f"Confidence: {conf}")

if __name__ == "__main__":
    test_calibration()
