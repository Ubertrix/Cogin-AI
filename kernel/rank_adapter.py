import numpy as np
from registry.config import SystemConfig

class RankAdapter:
    """
    Dynamic Rank Adaptation (The "Expansion" Protocol).
    When the neural complexity saturates, this adapter structurally maps all 
    pre-existing matrices into a higher-dimensional algebraic domain +1 natively.
    """
    def __init__(self):
        self.expansion_count = 0
        
    def expand_tensor(self, matrix, new_dim):
        """Maps an old matrix (A, B) gracefully into (A, new_dim) or (new_dim, new_dim)."""
        old_shape = matrix.shape
        
        # If it's a 1D vector (e.g. bias)
        if len(old_shape) == 1:
            if old_shape[0] >= new_dim:
                return matrix
            new_matrix = np.random.randn(new_dim) * 0.01
            new_matrix[:old_shape[0]] = matrix
            return new_matrix
            
        # If 2D matrix (e.g. weights)
        if len(old_shape) == 2:
            r, c = old_shape
            
            # If expanding both input and output structurally (e.g. SelfAttention queries)
            if r == c:
                new_r, new_c = new_dim, new_dim
            else:
                new_r = new_dim if r == SystemConfig.EMBEDDING_DIM else r
                new_c = new_dim if c == SystemConfig.EMBEDDING_DIM else c
                
            new_matrix = np.random.randn(new_r, new_c) * 0.01
            new_matrix[:r, :c] = matrix
            return new_matrix
            
        return matrix
