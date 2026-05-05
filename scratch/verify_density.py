import numpy as np
from kernel.ops.genesis import Genesis
from memory.long_term import EpistemicWeightMemory
from main import CogniPro
from registry.config import SystemConfig

print(f"--- Neural Density Verification (d_model={SystemConfig.EMBEDDING_DIM}) ---")

# 1. LayerNorm & GeLU Validation
genesis = Genesis()
v = np.random.randn(1, 1024) * 10.0 # High variance input
normalized = genesis.contextual_alignment("hello", v)
print(f"Input Dim: {v.shape} -> Result Dim: {normalized.shape}")
print(f"Normalized Mean: {np.mean(normalized):.4f} | Std: {np.std(normalized):.4f}")
assert normalized.shape == (1, 1024)
assert abs(np.mean(normalized)) < 0.1
assert abs(np.std(normalized) - 1.0) < 0.1

# 2. Epistemic Upscaler Test
memory = EpistemicWeightMemory(db_path="scratch/legacy_test_weights.npy")
legacy_vec = np.random.randn(512)
memory.save_knowledge("LegacyAnchor", "Legacy definition", vector=legacy_vec)

# Check if it upscaled
bundle = memory.knowledge_base["legacyanchor"]
# Dequantize should give 1024
upscaled = memory._dequantize(bundle["essence"], *bundle["bounds"])
print(f"Legacy 512-D Anchor -> Dequantized Shape: {upscaled.shape}")
assert upscaled.shape == (1024,)

# Retrieve with 1024-D query
query_1024 = np.random.randn(1, 1024)
res = memory.retrieve(query_vector=query_1024, top_k=1)
print(f"Retrieval status: {'Success' if res[0] else 'Failure'}")

# 3. Shape Stress Test (Full System)
cp = CogniPro()
print(f"CogniPro initialized with d_model={cp.d_model}")
resp, conf = cp.process("Hello Cogni Pro")
print(f"System Process Response: {resp[:50]}...")
print("Verification Complete.")
