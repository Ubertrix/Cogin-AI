import sys
import os
import time
import json
import numpy as np

# Add project path
sys.path.append('/home/ubuntu/Cogni_Pro/Cogni_Pro')

from brain_engine import CogniPro

def run_benchmark():
    print("--- COGNI PRO PERFORMANCE BENCHMARK ---")
    engine = CogniPro()
    
    test_cases = [
        {"query": "Write a Python function for binary search", "category": "Coding"},
        {"query": "Explain the concept of decorators in JavaScript", "category": "Coding"},
        {"query": "Solve the integral of x^2 dx", "category": "Math"},
        {"query": "What is the Pythagorean theorem?", "category": "Math"},
        {"query": "Explain Quantum Entanglement", "category": "Science"},
        {"query": "How does photosynthesis work?", "category": "Science"},
        {"query": "Who are you and what can you do?", "category": "Linguistic"},
        {"query": "مرحبا، كيف يمكنني تعلم البرمجة؟", "category": "Linguistic"}
    ]
    
    results = []
    
    for case in test_cases:
        query = case["query"]
        expected_cat = case["category"]
        
        print(f"\nTesting: {query[:50]}...")
        
        start_time = time.time()
        response, confidence = engine.process(query)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Simple heuristic for response quality
        word_count = len(response.split())
        quality_score = min(1.0, word_count / 10.0) # Expect at least 10 words for a good response
        
        results.append({
            "query": query,
            "expected_category": expected_cat,
            "latency": latency,
            "confidence": float(confidence),
            "word_count": word_count,
            "quality_score": quality_score
        })
        
        print(f"   > Latency: {latency:.4f}s | Confidence: {confidence:.2f} | Words: {word_count}")

    # Save results for analysis
    with open('/home/ubuntu/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nBenchmark complete. Results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
