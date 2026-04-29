import numpy as np

class SelfAttention:
    """Self-Attention Layer: Makes the model focus on the most important words in the sentence"""
    def __init__(self, d_model):
        self.d_model = d_model
        # Query, Key, and Value weights
        self.w_q = np.random.randn(d_model, d_model) * 0.01
        self.w_k = np.random.randn(d_model, d_model) * 0.01
        self.w_v = np.random.randn(d_model, d_model) * 0.01

    def forward(self, x, causal_mask=False):
        """
        x: Word matrix (seq_len, d_model)
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
            
        seq_len = x.shape[0]

        # Generate Q, K, V matrices
        queries = np.dot(x, self.w_q)
        keys = np.dot(x, self.w_k)
        values = np.dot(x, self.w_v)

        # Calculate Attention Scores
        # Multiply Q by Transposed K
        scores = np.dot(queries, keys.T) / np.sqrt(self.d_model)
        
        # Causal Masking: Prevent looking at "future" tokens
        if causal_mask and seq_len > 1:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            scores[mask] = -1e9 # Using -1e9 instead of -np.inf to avoid NaNs sometimes
        
        # Apply Softmax (simplified for small matrices)
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        
        if causal_mask and seq_len > 1:
             exp_scores[mask] = 0.0

        weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

        # Multiply weights by Values to get the final vector
        attention_output = np.dot(weights, values)
        
        # For dimensional stability, we strictly prevent 1D tensor squashing.
        # The output must guarantee a (seq_len, d_model) shape structure natively.
            
        return attention_output, weights
