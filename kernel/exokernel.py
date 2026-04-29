import numpy as np
import os
import gc
import time

class DSLAExokernel:
    """
    Cogni Decentralized-Exokernel (Sequential Stream Engine)
    Implementation of Dynamic Layered Stream Architecture (DSLA).
    Modified: Corrected paths to be relative to the project root.
    """
    def __init__(self, shard_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if shard_dir is None:
            self.shard_dir = os.path.join(base_dir, "kernel", "shards")
        else:
            self.shard_dir = shard_dir
            
        self.d_model = 1024
        self.active_context = np.zeros((1, self.d_model), dtype=np.float32)
        self.hardware_log_path = os.path.join(base_dir, "memory", "hardware_logs.bin")
        
        # Ensure shard directory exists
        os.makedirs(self.shard_dir, exist_ok=True)
        # Ensure memory directory exists for logs
        os.makedirs(os.path.dirname(self.hardware_log_path), exist_ok=True)

    def _pulse_layer(self, input_vec, layer_idx):
        """
        SEQUENTIAL READER LOGIC:
        1. Load layer from disk (Stream)
        2. Execute Matrix Ops (Pulse)
        3. Immediate Buffer Purge
        """
        shard_path = os.path.join(self.shard_dir, f"layer_{layer_idx}.npy")
        
        if not os.path.exists(shard_path):
            return input_vec # Fallback if layer doesn't exist
            
        # Stream from disk using memory map (Zero-copy intent)
        weights = np.load(shard_path, mmap_mode='r')
        
        # Computation (The Pulse)
        output = np.dot(input_vec, weights)
        
        # Purge (Weights reference cleared, mmap handle released on deletion)
        del weights
        return output

    def stream_inference(self, input_vector):
        """
        Processes input through the sequential stream of layers.
        Adheres to Cogni Memory Formula.
        """
        state = input_vector
        if not os.path.exists(self.shard_dir):
            return state
            
        num_layers = len([f for f in os.listdir(self.shard_dir) if f.endswith(".npy")])
        
        if num_layers == 0:
            return state
            
        print(f"[DSLA Engine] Starting Sequential Pulse (Layers: {num_layers})")
        start_time = time.time()
        
        for i in range(num_layers):
            # The "Reader's Eye" movement
            state = self._pulse_layer(state, i)
            
            # Context Externalization (Simulated update to hardware logs)
            self._log_to_hardware(state, i)
            
            # Explicit Memory Purge for the next layer
            gc.collect()
            
        duration = time.time() - start_time
        print(f"[DSLA Engine] Stream Complete. Throughput: {num_layers/duration:.2f} layers/sec")
        return state

    def _log_to_hardware(self, state, layer_idx):
        """Bypasses KV-Cache. Writes state directly to hardware-mapped logs."""
        try:
            with open(self.hardware_log_path, "ab") as f:
                f.write(state.tobytes())
        except:
            pass
            
    def reset_hardware_logs(self):
        if os.path.exists(self.hardware_log_path):
            os.remove(self.hardware_log_path)

if __name__ == "__main__":
    exokernel = DSLAExokernel()
    test_input = np.random.randn(1, 1024).astype(np.float32)
    output = exokernel.stream_inference(test_input)
    print(f"Final State Norm: {np.linalg.norm(output)}")
