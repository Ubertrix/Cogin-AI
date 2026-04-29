import sys
import os
import numpy as np

sys.path.append('/Users/cela/Desktop/Cogni_Pro')

def build_dsla():
    print("=== COGNI DSLA BUILD ===")
    
    # Step 1: Initialize CogniPro to extract real weights
    print("\n[1/4] Loading CogniPro model...")
    try:
        from brain_engine import CogniPro
        cogni = CogniPro()
        print("  ✓ Model loaded")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return
    
    # Step 2: Export layers to DSLA shards
    print("\n[2/4] Exporting model layers to shards...")
    shard_dir = '/Users/cela/Desktop/Cogni_Pro/kernel/shards'
    os.makedirs(shard_dir, exist_ok=True)
    
    # Clear old shards
    for f in os.listdir(shard_dir):
        os.remove(os.path.join(shard_dir, f))
    
    shard_idx = 0
    
    # Inference Head (transposed to d_model x vocab_size for dot product)
    inf_head = cogni.inference_head
    # weights are (vocab_size, d_model), transpose to (d_model, vocab_size)
    head_weights = inf_head.weights.T  # Now (d_model, vocab_size)
    np.save(os.path.join(shard_dir, f'layer_{shard_idx}.npy'), head_weights)
    print(f"  ✓ Layer {shard_idx} (Inference Head Projection): {head_weights.shape}")
    shard_idx += 1
    
    print(f"  Total shards: {shard_idx}")
    
    # Step 3: Verify Exokernel
    print("\n[3/4] Verifying DSLA Exokernel...")
    try:
        from kernel.exokernel import DSLAExokernel
        exokernel = DSLAExokernel()
        print("  ✓ Exokernel initialized")
    except Exception as e:
        print(f"  ✗ Exokernel error: {e}")
        return
    
    # Step 4: Test stream inference
    print("\n[4/4] Testing stream inference...")
    try:
        test_input = np.random.randn(1, 1024).astype(np.float32)
        output = exokernel.stream_inference(test_input)
        print(f"  ✓ Stream complete. Output norm: {np.linalg.norm(output):.4f}")
    except Exception as e:
        print(f"  ✗ Stream failed: {e}")
        return
    
    print("\n=== BUILD COMPLETE ===")

if __name__ == '__main__':
    build_dsla()
