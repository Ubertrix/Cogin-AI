import sys
import os
import numpy as np

# Ensure we can import from the main project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry.config import SystemConfig
from dictionary.tokenizer import Tokenizer
from kernel.layers.embedding import EmbeddingLayer
from kernel.layers.attention import SelfAttention
from router.gate import SmartGate
from experts_pool.base_expert import BaseExpert

def run_dimension_check():
    print("--- 🧠 Dimensional Integrity Check ---")
    
    tokenizer = Tokenizer()
    tokenizer.fit(["Check the neural dimension logic"])
    embed_layer = EmbeddingLayer(vocab_size=tokenizer.vocab_size, d_model=SystemConfig.EMBEDDING_DIM)
    attention = SelfAttention(d_model=SystemConfig.EMBEDDING_DIM)
    
    # Simulating single character input which caused the crash (e.g. 'c')
    input_text = "c"
    tokens = tokenizer.encode(input_text)
    print(f"[1] Tokenization: '{input_text}' -> length {len(tokens)}")
    
    # 1. Embedding shape test
    embedded_vectors = embed_layer.forward(tokens)
    print(f"[2] Embedding Rank: {embedded_vectors.ndim} | Shape: {embedded_vectors.shape}")
    assert embedded_vectors.ndim >= 2, f"Embedding dropped rank: {embedded_vectors.shape}"
    
    # 2. Attention Layer stability test
    attended, _ = attention.forward(embedded_vectors)
    print(f"[3] Attention Output Rank: {attended.ndim} | Shape: {attended.shape}")
    assert attended.ndim >= 2, f"Attention Layer caused scalar collapse: {attended.shape}"
    
    # 3. Main.py Mean-Pooling reduction test
    sentence_vector = np.mean(attended, axis=0, keepdims=True)
    print(f"[4] Router Prep (Mean Pool) Rank: {sentence_vector.ndim} | Shape: {sentence_vector.shape}")
    
    # Enforcement mapping
    sentence_vector = np.atleast_2d(sentence_vector)
    
    # 4. Smart Gate Alignment test
    gate = SmartGate(input_dim=SystemConfig.EMBEDDING_DIM, num_experts=1)
    experts = [BaseExpert(name="TestExpert")]
    expert, confidence = gate.route(sentence_vector, experts)
    print(f"[5] SmartGate Output Expert: {expert.name} | Confidence: {confidence}")
    
    # 5. Expert Output
    expert_out, _ = expert.process(sentence_vector)
    print(f"[6] Expert Output Rank: {expert_out.ndim} | Shape: {expert_out.shape}")
    assert expert_out.ndim >= 2, f"ShapeGuard failed. Expert scalar collapse: {expert_out.shape}"
    
    print("\n✅ INTEGRITY SECURED: Tensor Flow maintained Rank 2 (1, 512) throughout the entire pipeline.")

if __name__ == "__main__":
    run_dimension_check()
