import numpy as np

class Genesis:
    def __init__(self, stability_factor=0.95):
        self.stability = stability_factor
        print("Genesis Module: Ready with LayerNorm & GeLU Activation (1024-D)")

    def layer_normalization(self, x, eps=1e-6):
        """Implement Layer Normalization: keeps mean at 0 and std at 1.0."""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + eps
        return (x - mean) / std

    def gelu(self, x):
        """Gaussian Error Linear Unit (Approximated)."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def spawn_cell(self, input_dim, output_dim):
        return self.layer_normalization(np.random.randn(input_dim, output_dim) * 0.01)

    def evaluate_inference(self, logits, chosen_token_id, confidence, penalty=10.0, chaos_factor=0.01):
        """
        Self-Correction Loop with Entropy Seeding.
        """
        entropy_noise = np.random.randn(*logits.shape) * chaos_factor
        logits = logits + entropy_noise

        if confidence < self.stability:
            corrected_logits = np.copy(logits)
            corrected_logits[chosen_token_id] -= penalty 
            return corrected_logits
            
        return None

    def contextual_alignment(self, input_text, vector_space, cluster_kw="hello", bias_strength=0.1):
        """
        Re-aligns the vector space context dynamically with LayerNorm stabilization.
        """
        if cluster_kw in input_text.lower():
            cluster_bias = np.random.randn(*vector_space.shape) * bias_strength
            cluster_bias = np.abs(cluster_bias) 
            aligned = vector_space + cluster_bias
            # Enforce Neuron Normalization after each Genesis Step
            return self.layer_normalization(self.gelu(aligned))
        
        return self.layer_normalization(self.gelu(vector_space))
