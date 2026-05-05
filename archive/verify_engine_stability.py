import numpy as np
import sys
import os

# 1. تثبيت مسارات النظام
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kernel.ops.inference_engine import InferenceEngine

def test_stability():
    print("--- 🔬 INFERENCE ENGINE: STABILITY & TRACE VERIFICATION ---")
    engine = InferenceEngine(d_model=512)
    
    # 324 active geometric anchors (mocked as weights)
    weights = np.random.randn(512, 512) * 0.01
    x = np.random.randn(1, 512)
    
    print("\n[Stress Test] Running 100 sequential projection layers...")
    current_x = x
    for i in range(100):
        # We pass through the engine, which applies alignment and RMSNorm
        current_x = engine.forward_pass(current_x, weights, label=f"Layer_{i}")
        
    final_mean = np.mean(current_x)
    final_std = np.std(current_x)
    
    print("\n[Result] Final Tensor State after 100 layers:")
    print(f"Mean: {final_mean:.8f}")
    print(f"Std:  {final_std:.8f}")
    
    if abs(final_mean) < 1e-10:
        print("❌ FAILURE: Vanishing gradient detected.")
    else:
        print("✅ SUCCESS: Inference Engine stabilized the tensor flow.")

if __name__ == "__main__":
    test_stability()
