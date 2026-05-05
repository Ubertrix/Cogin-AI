import sys
import os
import numpy as np

# Ensure we can import CogniPro
sys.path.append(os.getcwd())
from main import CogniPro

def test_cohesion():
    print("--- 🧪 NEURAL COHESION TEST SUITE ---")
    p = CogniPro()
    
    # Test 1: Math Integrity
    print("\n[Test 1] Query: '1+1' (Logic Integrity)")
    response = p.process("1+1")
    print(f"Response: {response}")
    
    # Test 2: Coding LoopBreak
    print("\n[Test 2] Query: 'print' (Loop & Repetition Check)")
    response = p.process("print")
    print(f"Response: {response}")

    # Test 3: Memory Anchor
    print("\n[Test 3] Query: 'hello' (Bridge Reinforcement)")
    response = p.process("hello")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_cohesion()
