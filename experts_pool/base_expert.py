import numpy as np
from kernel.shape_guard import shape_guard

class BaseExpert:
    def __init__(self, name="Generalist", input_dim=1024, output_dim=1024):
        self.name = name
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.dropout_rate = 0.1 # 10% Dropout during training

    @shape_guard
    def process(self, x, input_text=""):
        """العملية الحسابية (1024-D Feed-Forward)"""
        x = np.atleast_2d(x)
        activity = np.dot(x, self.weights) + self.bias
        return activity, None

    def distill(self, input_vector, target_vector, lr=0.01, sparsity_threshold=1e-4):
        """
        Synaptic Update (Hebbian Learning) with 10% Dropout.
        """
        input_vector = np.atleast_2d(input_vector)
        target_vector = np.atleast_2d(target_vector)
        
        # Apply Dropout mask to the input during training phase
        mask = (np.random.rand(*input_vector.shape) > self.dropout_rate).astype(np.float32)
        dropped_input = input_vector * mask
        
        prediction = np.dot(dropped_input, self.weights)
        error = target_vector - prediction
        
        if np.max(np.abs(error)) < sparsity_threshold:
             return
             
        delta_w = np.dot(dropped_input.T, error)
        self.weights += lr * delta_w
        
        # Self-normalize
        max_norm = np.max(np.abs(self.weights))
        if max_norm > 1e3:
             self.weights /= (max_norm / 1e3)
