import sys
import os
import time
import numpy as np
import json

# Add project path
sys.path.append('/home/ubuntu/Cogni_Pro/Cogni_Pro')

from brain_engine import CogniPro

class ArchitecturalMonitor:
    """Monitors the internal flow of the Cogni Pro architecture."""
    def __init__(self):
        self.logs = []

    def log_event(self, layer, action, details):
        event = {
            "timestamp": time.time(),
            "layer": layer,
            "action": action,
            "details": details
        }
        self.logs.append(event)
        print(f"   [ARCH_MONITOR] {layer} -> {action} | {details}")

def start_interactive_session():
    print("\n" + "="*50)
    print("   COGNI PRO INTERACTIVE ARCHITECTURAL SESSION")
    print("="*50)
    
    engine = CogniPro()
    monitor = ArchitecturalMonitor()
    
    print("\nSystem is ready. Type 'exit' to end the session.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'خروج']:
            break
            
        print("\n--- Processing Architecture Flow ---")
        
        # Monitor Step 1: Tokenization
        start = time.time()
        tokens = engine.tokenizer.encode(user_input)
        monitor.log_event("Tokenizer", "Encoding", f"Tokens: {len(tokens)}")
        
        # Monitor Step 2: Embedding & Attention
        embeddings = engine.embedding_layer.forward(np.array(tokens))
        monitor.log_event("EmbeddingLayer", "Vectorization", f"Shape: {embeddings.shape}")
        
        attended, _ = engine.attention.forward(embeddings)
        monitor.log_event("SelfAttention", "Contextualization", "Attention weights applied")
        
        # Monitor Step 3: Routing
        input_vector = np.mean(attended, axis=0, keepdims=True)
        expert, confidence = engine.router.route(input_vector, engine.experts, input_text=user_input)
        monitor.log_event("SmartGate", "Routing", f"Target: {expert.name} | Confidence: {confidence:.2f}")
        
        # Monitor Step 4: Knowledge Retrieval
        ret_data = engine.long_term.retrieve(query_text=user_input, query_vector=input_vector, filter_category=expert.name)
        found_knowledge = "Yes" if (isinstance(ret_data, tuple) and ret_data[4]) else "No"
        monitor.log_event("LongTermMemory", "Retrieval", f"Knowledge Found: {found_knowledge}")
        
        # Monitor Step 5: Final Processing
        response, final_conf = engine.process(user_input)
        end = time.time()
        
        monitor.log_event("BrainEngine", "Synthesis", f"Total Latency: {end-start:.4f}s")
        
        print(f"\nCogni Pro: {response}")
        print(f"(Confidence: {final_conf})")

    # Save logs
    with open('/home/ubuntu/architectural_logs.json', 'w') as f:
        json.dump(monitor.logs, f, indent=4)
    print("\nSession ended. Architectural logs saved.")

if __name__ == "__main__":
    start_interactive_session()
