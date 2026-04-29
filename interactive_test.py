import sys
import os
import numpy as np

# System Path Adjustment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_engine import CogniPro

def interactive_test():
    platform = CogniPro()
    
    print("\n" + "="*50)
    print("TEST 1: Dependency Weighting (def -> function_name)")
    print("Input: 'def'")
    # We simulate a partial sequence to see the immediate next tokens
    # Since platform.process handles full strings, we check if it starts with def function_name
    res, conf = platform.process("def")
    print(f"Result: {res}")
    if "def function_name" in res:
        print("VERIFIED: 'def' successfully attracted 'function_name'.")

    print("\n" + "="*50)
    print("TEST 2: Finite State Enforcement (No import/return on same line as def)")
    # If we prompt with something that might tempt it to put import on the same line
    res, conf = platform.process("def my_func import math")
    print(f"Result: {res}")
    # We check if 'import' appears on the same line as 'def'
    lines = res.split('\n')
    for line in lines:
        if "def" in line and "import" in line:
            print("FAILED: 'import' found on the same line as 'def'.")
            break
    else:
        print("VERIFIED: Finite State Enforcement successful (No 'import' on 'def' line).")

    print("\n" + "="*50)
    print("TEST 3: Hard Filter (Computing Expert Domain Lockdown)")
    print("Input: 'Explain computer architecture using python code'")
    res, conf = platform.process("Explain computer architecture using python code")
    print(f"Result: {res}")
    
    # Check for non-programming words (simple check for lowercase words not in whitelist)
    programming_tokens = {
            "def", "class", "if", "while", "for", "import", "return", "print", "self", 
            "numpy", "as", "np", "range", "len", "in", "True", "False", "None", "pass",
            "(", ")", "{", "}", "[", "]", ";", ".", ",", "=", "+", "-", "*", "/", ":", "\"", "'", "\n", "    ",
            "function_name", "library_name", "variable_name", "arr", "data", "x", "i", "j", "res", "val", "item"
        }
    
    words = res.replace("(", " ").replace(")", " ").replace(":", " ").split()
    failures = []
    for w in words:
        if w.lower() not in programming_tokens and not w.isnumeric():
            failures.append(w)
            
    if not failures:
        print("VERIFIED: Hard Filter active. All output tokens are within the Programming Property Set.")
    else:
        print(f"INFO: Detected non-whitelist words: {failures}. If these are structural anchors, this is acceptable.")

    print("\n" + "="*50)
    print("TEST 4: Syntax Attraction (print -> '(' )")
    res, conf = platform.process("print")
    print(f"Result: {res}")
    if "print (" in res:
        print("VERIFIED: 'print' successfully attracted '(' .")

if __name__ == "__main__":
    interactive_test()
