import numpy as np

class GenerativeHead:
    """
    The Neural Generative Engine (v5.0).
    Simplified for maximum stability.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def apply_softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-9)

    def sample(self, logits, **kwargs):
        # Simple Greedy/Top-K Sampling
        probs = self.apply_softmax(logits)
        
        # Repetition Penalty
        token_history = kwargs.get('token_history', [])
        if token_history:
            for tid in token_history[-10:]:
                probs[tid] *= 0.1
            probs /= np.sum(probs)

        # Pick the best token
        token_id = np.argmax(probs)
        confidence = float(probs[token_id])
        
        return token_id, confidence

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
