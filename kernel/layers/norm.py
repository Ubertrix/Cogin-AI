import numpy as np

class RMSNorm:
    """
    Root Mean Square Layer Normalization.
    Prevents vanishing/exploding gradients by scaling tensors based on their RMS.
    formula: y = (x / sqrt(mean(x^2) + eps)) * gamma
    """
    def __init__(self, d_model, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones(d_model) # Learnable parameter (logic placeholder)
        
    def forward(self, x):
        # Calculate RMS
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        # Scale and apply gamma
        return (x / rms) * self.gamma
