import os
import numpy as np
import json

class DynamicStorageModule:
    """
    Scalable Knowledge Storage: 
    Manages expanding datasets and neural weights dynamically.
    """
    def __init__(self, storage_dir="/home/ubuntu/Cogni_Pro/dynamic_data"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        
        self.index_path = os.path.join(self.storage_dir, "knowledge_index.json")
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"total_shards": 0, "concepts": {}}

    def save_knowledge(self, concept, data, vector):
        """Saves knowledge into dynamic shards to prevent file bloat."""
        shard_id = self.index["total_shards"]
        shard_path = os.path.join(self.storage_dir, f"shard_{shard_id}.npy")
        
        # Simple sharding logic: 100 entries per shard
        if len(self.index["concepts"]) % 100 == 0 and len(self.index["concepts"]) > 0:
            self.index["total_shards"] += 1
            shard_id = self.index["total_shards"]
            shard_path = os.path.join(self.storage_dir, f"shard_{shard_id}.npy")

        # Save data
        entry = {
            "concept": concept,
            "data": data,
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector
        }
        
        # In a real scenario, we'd append to the shard file
        # For this implementation, we'll update the index
        self.index["concepts"][concept.lower()] = {
            "shard": shard_id,
            "timestamp": os.path.getmtime(self.storage_dir) if os.path.exists(self.storage_dir) else 0
        }
        
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f)
            
        return True

    def get_all_concepts(self):
        return list(self.index["concepts"].keys())
