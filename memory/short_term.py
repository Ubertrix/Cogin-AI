from collections import defaultdict
import numpy as np

class ShortTermMemory:
    """ذاكرة الجلسة الحالية (Context Window) مع دعم الربط الاحتمالي (Probabilistic KV-Linkage)"""
    def __init__(self, max_capacity=10, d_model=1024):
        self.capacity = max_capacity
        self.d_model = d_model
        self.history = [] # list to save recent interactions
        
        # Pure NumPy GRU Weights for Context Buffer
        self.W_z = np.random.randn(d_model * 2, d_model) * 0.01
        self.W_r = np.random.randn(d_model * 2, d_model) * 0.01
        self.W_h = np.random.randn(d_model * 2, d_model) * 0.01
        
        # KV-Linkage tracking for transition probabilities
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.recent_tokens = [] # keep track of real-time generated tokens for bias

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def add_interaction(self, user_input, system_output, input_vector=None):
        """Add a new interaction with vector for GRU flow"""
        if len(self.history) >= self.capacity:
            self.history.pop(0)
        self.history.append({
            "user": user_input, 
            "system": system_output,
            "vector": np.atleast_2d(input_vector) if input_vector is not None else np.zeros((1, self.d_model))
        })

    def get_cumulative_hidden_state(self):
        """
        Context Buffer Implementation:
        Converts sliding window buffer (last 5 inputs) into a cumulative hidden state (1, 512).
        Uses pure NumPy GRU logic.
        """
        h_t = np.zeros((1, self.d_model))
        for item in self.history:
            x_t = item["vector"]
            if x_t.shape[-1] != self.d_model:
                x_t = np.zeros((1, self.d_model))
                
            # Connect x_t and h_{t-1}
            concat = np.concatenate([x_t, h_t], axis=-1)  # (1, d_model * 2)
            
            # GRU Equations
            z_t = self._sigmoid(np.dot(concat, self.W_z))
            r_t = self._sigmoid(np.dot(concat, self.W_r))
            
            concat_reset = np.concatenate([x_t, r_t * h_t], axis=-1)
            h_tilde = np.tanh(np.dot(concat_reset, self.W_h))
            
            # Hidden state update
            h_t = (1 - z_t) * h_t + z_t * h_tilde
            
        return h_t

    def get_recent_context(self):
        return self.history

    def update_kv_linkage(self, token_sequence):
        """
        Record transition probabilities between concepts/tokens.
        Expected token_sequence is a list of token IDs (integers).
        """
        for i in range(len(token_sequence) - 1):
            current_token = token_sequence[i]
            next_token = token_sequence[i+1]
            self.transition_counts[current_token][next_token] += 1
            
        # keep last known tokens for immediate bias
        if token_sequence:
            # Keep a longer autoregressive buffer for context injection (last 10 tokens)
            self.recent_tokens = token_sequence[-10:]

    def get_inference_bias(self, vocab_size, current_token_id=None):
        """
        Calculates Probabilistic KV-Linkage bias.
        Boosts probabilities of tokens that historically follow the current token.
        """
        bias = np.zeros(vocab_size)
        
        # If no specific token given, use the most recent token from memory
        if current_token_id is None and self.recent_tokens:
            current_token_id = self.recent_tokens[-1]
            
        if current_token_id is not None and current_token_id in self.transition_counts:
            transitions = self.transition_counts[current_token_id]
            total_transitions = sum(transitions.values())
            
            if total_transitions > 0:
                for next_token, count in transitions.items():
                    if next_token < vocab_size:
                        # Log-based boost strategy based on frequency probability
                        prob = count / total_transitions
                        # +2.0 base boost for linked tokens, scaling with log probability
                        bias[next_token] = np.log(prob + 1e-5) + 2.0 
                        
        return bias
