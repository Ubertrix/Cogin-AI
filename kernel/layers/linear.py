import numpy as np

class LinearLayer:
    """Fully Connected Layer"""
    def __init__(self, input_dim, output_dim):
        # Use He Initialization to ensure numerical stability
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        
        # Variables to store layer state for backpropagation (Gradients)
        self.last_input = None

    def forward(self, x):
        """Forward propagation: Thinking"""
        self.last_input = x
        return np.dot(x, self.weights) + self.bias
