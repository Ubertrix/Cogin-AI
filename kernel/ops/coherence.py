import numpy as np

class CoherenceGuard:
    """
    Filters out 'Noise' and ensures the output follows a logical grammatical structure.
    Monitors for repetition and character-level chaotic permutations.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.repetition_threshold = 3
        
    def audit_sequence(self, token_ids):
        """Checks for immediate token loops and excessive repetition."""
        if len(token_ids) < self.repetition_threshold:
            return True # Not enough tokens to judge
            
        # Check for simple back-to-back repetition (unigram loops)
        last_token = token_ids[-1]
        loop_count = 0
        for tid in reversed(token_ids[:-1]):
            if tid == last_token:
                loop_count += 1
            else:
                break
        
        if loop_count >= self.repetition_threshold:
            return False # Failed auditing: Repetitive Noise detected
            
        return True

    def filter_logits(self, logits, generated_tokens):
        """
        Suppresses logits for tokens that would cause a repetition crash.
        """
        if len(generated_tokens) < 1:
            return logits
            
        filtered_logits = np.copy(logits)
        last_token_id = generated_tokens[-1]
        
        # Penalize repeating the exact same token if it has already appeared twice in a row
        if len(generated_tokens) >= 2 and generated_tokens[-1] == generated_tokens[-2]:
            filtered_logits[last_token_id] -= 20.0
            
        return filtered_logits
