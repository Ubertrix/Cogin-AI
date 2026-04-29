import numpy as np
import os

class EmbeddingLayer:
    """Convert word IDs to vectors - DIRECT LOADING from brain_weights"""
    def __init__(self, vocab_size, d_model):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Initialize embeddings
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.01
        
        # DIRECT LOADING: Try to load from brain_weights
        self._load_from_brain()

    def _load_from_brain(self):
        """Load embeddings directly from brain_weights.npy"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        brain_path = os.path.join(base_dir, "brain_weights.npy")
        
        if not os.path.exists(brain_path):
            return
            
        try:
            data = np.load(brain_path, allow_pickle=True).item()
            concepts = list(data.keys())
            
            # Build embeddings from brain weights
            for i, concept in enumerate(concepts[:self.vocab_size]):
                entry = data[concept]
                if "essence" in entry:
                    vec = entry["essence"]
                    if hasattr(vec, 'shape') and len(vec.shape) > 0:
                        vec_flat = vec.flatten()[:self.d_model]
                        if i < self.vocab_size:
                            self.embeddings[i] = vec_flat
            
            print(f"   [Embedding] Loaded {len(concepts)} embeddings from brain_weights")
        except Exception as e:
            print(f"   [Embedding] Using random init: {e}")

    def forward(self, tokens):
        # Fetch vectors for words in the sentence
        # Ensure index is within bounds (Safety Check)
        valid_tokens = [t if t < self.embeddings.shape[0] else 1 for t in tokens]
        return self.embeddings[valid_tokens]

    def grow(self, new_vocab_size):
        """Expand embedding matrix to accommodate new words without losing old ones"""
        if new_vocab_size <= self.embeddings.shape[0]:
            return
            
        diff = new_vocab_size - self.embeddings.shape[0]
        new_weights = np.random.randn(diff, self.d_model) * 0.01
        self.embeddings = np.vstack([self.embeddings, new_weights])
        self.vocab_size = new_vocab_size
        print(f"Embedding Layer expanded to {new_vocab_size} tokens.")
