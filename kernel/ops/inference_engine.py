import numpy as np
from kernel.layers.norm import RMSNorm
from kernel.ops.tracker import NeuralTracker

class InferenceEngine:
    """
    The Optimized Modular Inference Engine.
    Handles matrix alignment, normalization, and tracing for the Cogni Pro core.
    Enforces strict (1, 512) tensor flow.
    """
    def __init__(self, d_model=512):
        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.tracker = NeuralTracker(name="Genesis")
        
    def forward_pass(self, x, weights, bias=None, label="Projection"):
        """
        Robust Forward Pass:
        1. Aligns input to (1, d_model).
        2. Applies Dot Product with weights (d_model, N).
        3. Applies Bias (1, N) if provided.
        4. Applies Normalization to prevent collapsing.
        5. Logs diagnostics to the Neural Map Tracker.
        """
        # --- 1. Matrix Alignment ---
        self.tracker.trace(x, f"{label}_In")
        
        # Ensure we are (1, d_model)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.shape[0] > 1:
            x = x[-1:] # Take latest context
            
        if x.shape[1] != self.d_model:
            # Emergency correction for dimensionality mismatch
            aligned_x = np.zeros((1, self.d_model))
            flat = x.flatten()
            size = min(len(flat), self.d_model)
            aligned_x[0, :size] = flat[:size]
            x = aligned_x

        # --- 2. Computation ---
        # x is (1, 512), weights is (512, N)
        output = np.dot(x, weights)
        
        if bias is not None:
            output += bias
            
        # --- 3. Normalization (Vanishing Gradient Protection) ---
        # Note: RMSNorm is applied to the output feature dimension
        if output.shape[-1] == self.d_model:
            output = self.norm.forward(output)
            
        # --- 4. Diagnostics ---
        self.tracker.trace(output, f"{label}_Out")
        
        return output
