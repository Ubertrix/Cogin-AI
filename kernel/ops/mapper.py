import numpy as np

class TokenMapper:
    """
    Semantic Projection Layer: Maps high-dimensional latent vectors back to the
    token space using Cosine Similarity instead of simple dot-products.
    This ensures we pick the most 'meaningful' word according to geometric truth.
    """
    def __init__(self, embeddings_matrix):
        # embeddings_matrix is (vocab_size, d_model)
        self.embeddings = embeddings_matrix
        self._norm_embeddings()

    def _norm_embeddings(self):
        """Pre-calculate norms for fast cosine similarity."""
        self.e_norms = np.linalg.norm(self.embeddings, axis=1) + 1e-9

    def project(self, context_vector, boost_vector=None, suppress_numeric=False, tokenizer=None):
        """
        Calculates cosine similarity between context_vector (1, d_model) 
        and all tokens (V, d_model), with optional noise suppression.
        """
        v = np.squeeze(context_vector)
        v_norm = np.linalg.norm(v) + 1e-9
        
        # Sim = (H . W) / (|H| * |W|)
        dots = np.dot(self.embeddings, v)
        similarities = dots / (self.e_norms * v_norm)
        
        # Scale similarities to logit-friendly range
        logits = similarities * 50.0
        
        # Suppress numeric IDs if requested (Semantic Noise Clean-up)
        if suppress_numeric and tokenizer:
            for tid in range(tokenizer.vocab_size):
                if tokenizer.decode([tid]).isdigit():
                    logits[tid] -= 100.0

        if boost_vector is not None:
            logits += boost_vector
            
        return logits
