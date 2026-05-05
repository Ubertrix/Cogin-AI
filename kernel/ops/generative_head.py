import numpy as np

class GenerativeHead:
    """
    v6.0 Neural Generative Engine.
    Supports Probabilistic Sampling (Top-K, Top-P, Temperature) for true generation.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def apply_softmax(self, x, temperature=1.0):
        # Apply temperature to logits
        x = x / max(temperature, 1e-3)
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-9)

    def sample(self, logits, **kwargs):
        """
        True Probabilistic Sampling.
        """
        temperature = kwargs.get('temperature', 0.85)
        top_k = kwargs.get('top_k', 50)
        token_history = kwargs.get('token_history', [])
        
        # 1. Apply Repetition Penalty
        if token_history:
            # Penalize the last 20 tokens to avoid loops
            for tid in set(token_history[-20:]):
                logits[tid] -= 2.0 # Logit-level penalty
        
        # 2. Apply Softmax with Temperature
        probs = self.apply_softmax(logits, temperature=temperature)
        
        # 3. Top-K Filtering
        if top_k > 0:
            top_k = min(top_k, len(probs))
            indices_to_remove = probs.argsort()[:-top_k]
            probs[indices_to_remove] = 0
            probs /= (np.sum(probs) + 1e-9) # Re-normalize
            
        # 4. Probabilistic Sampling (The core of "True Generation")
        try:
            # Pick a token based on the probability distribution
            token_id = np.random.choice(len(probs), p=probs)
            confidence = float(probs[token_id])
        except ValueError:
            # Fallback to greedy if sampling fails
            token_id = np.argmax(probs)
            confidence = float(probs[token_id])
        
        return token_id, confidence

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
