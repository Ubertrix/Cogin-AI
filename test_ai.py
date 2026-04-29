#!/usr/bin/env python3
import sys
import os
sys.path.append("/Users/cela/Desktop/Cogni_Pro")

from brain_engine import CogniPro

platform = CogniPro()
platform.train_knowledge(["System startup"])

print("\n--- Testing AI Engine Responses ---\n")

tests = [
    "hello how are you",
    "what is ai",
    "tell me about the world",
    "who are you",
    "what is python",
    "how are you feeling today",
    "explain quantum physics",
    "what do you think about life",
    "write a python function",
    "what is love",
    "bye"
]

for text in tests:
    print(f"\nYou > {text}")
    response, conf = platform.process(text)
    print(f"Cogni > {response}\n")
    print(f"   [Confidence: {conf:.2f}]")
    print("-" * 50)

print("\n--- Test Complete ---")