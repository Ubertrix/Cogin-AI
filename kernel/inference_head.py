import numpy as np
from kernel.ops.mapper import TokenMapper
from kernel.ops.inference_engine import InferenceEngine
from kernel.ops.generative_head import GenerativeHead

class InferenceHead:
    """
    The Predictive Engine (Generative Head). 
    Transforms context vectors from experts/attention layers into probabilities across the vocabulary
    using Autoregressive Prediction and Semantic Inference.
    """
    def __init__(self, d_model, vocab_size, tokenizer, embeddings_matrix=None):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Weight tying: share weights with the embedding layer to save memory
        if embeddings_matrix is not None:
            self.weights = embeddings_matrix
        else:
            # Fallback if no embedding matrix is provided
            self.weights = np.random.randn(vocab_size, d_model) * 0.01
            
        self.engine = InferenceEngine(d_model)
        self.mapper = TokenMapper(self.weights)
        self.scribe = GenerativeHead(tokenizer)

    def update_vocab_size(self, new_vocab_size, embeddings_matrix):
        """Allows dynamically growing the vocab without losing the bound weights."""
        self.vocab_size = new_vocab_size
        self.weights = embeddings_matrix
        self.mapper = TokenMapper(self.weights)

    def Safe_Project(self, context_vector):
        """
        Dynamic Reshaping Logic: Automatically reshapes vectors to (1, d_model).
        Ensures context_vector is properly aligned before matrix multiplication.
        """
        # Catch single token index or scalar errors
        if np.isscalar(context_vector) or np.size(context_vector) == 1:
            print(f"   [InferenceHead Warning] Alignment Error: Received scalar {context_vector}. Expanding to (1, {self.d_model}).")
            baseline = np.zeros((1, self.d_model))
            baseline[0, 0] = float(np.squeeze(context_vector))
            return baseline
            
        context_vector = np.atleast_2d(context_vector)
        
        if context_vector.shape == (self.d_model, 1):
            context_vector = context_vector.T
            
        if context_vector.shape[-1] != self.d_model:
            print(f"   [InferenceHead Warning] Dimensional mismatch {context_vector.shape}. Restoring standard d_model space.")
            clean_vector = np.zeros((1, self.d_model))
            flat = context_vector.flatten()
            size = min(len(flat), self.d_model)
            clean_vector[0, :size] = flat[:size]
            return clean_vector
            
        return context_vector[-1:] # Always return (1, d_model)

    def decode_logits(self, context_vector, inference_bias=None, suppress_numeric=False, tokenizer=None):
        """
        The Decoder Bridge: Maps the output of the Attention layer back to the 
        Multi-Language Dictionary (the 9,248 tokens) using Semantic Projection.
        Ensures 'Syntactic Synergy' between the attention state and token space.
        """
        return self.forward(context_vector, inference_bias, projection="semantic", suppress_numeric=suppress_numeric, tokenizer=tokenizer)

    def forward(self, context_vector, inference_bias=None, projection="semantic", suppress_numeric=False, tokenizer=None, anchor_vector=None):
        """
        v5.0 Inference Head:
        Clean logit projection from context to vocabulary space.
        """
        H = self.Safe_Project(context_vector)

        # Projection
        if self.vocab_size > 1500:
             logits = np.dot(H, self.weights.T).squeeze(0)
        else:
             logits = np.dot(H, self.weights.T).squeeze(0)

        # Normalise
        scale = np.std(logits)
        if scale > 1e-9:
            logits = logits / scale

        # Add inference bias from memory
        if inference_bias is not None:
            try:
                logits = logits + np.squeeze(inference_bias)[:self.vocab_size]
            except Exception:
                pass

        # Anchor Boost from memory
        if anchor_vector is not None:
            A = np.atleast_1d(np.squeeze(anchor_vector))
            if A.shape[0] == self.d_model:
                w_norms = np.linalg.norm(self.weights, axis=1) + 1e-9
                a_norm = np.linalg.norm(A) + 1e-9
                neighborhood_boost = np.dot(self.weights, A) / (w_norms * a_norm)
                # Reduce anchor boost to avoid domination during decoding
                logits = logits + (neighborhood_boost * 2.0)

        return logits

    def sample(self, *args, **kwargs):
        """Delegates to the GenerativeHead (Neural Scribe)."""
        return self.scribe.sample(*args, **kwargs, weights=self.weights, anchors=getattr(self, 'anchors', None))
