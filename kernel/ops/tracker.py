import numpy as np

# Global Debug Mode: Set to True to see all Neural Map traces
DEBUG_MODE = False

class NeuralTracker:
    """
    The Neural Map Tracker: Monitors tensor health during inference.
    Logs Shape, Mean, and Standard Deviation to detect 'Scalar Collapse'.
    In production mode (DEBUG_MODE=False), only critical warnings are shown.
    """
    def __init__(self, name="Core"):
        self.name = name
        self.step_count = 0

    def trace(self, tensor, layer_label="Layer"):
        self.step_count += 1
        
        # Calculate stats
        shape = tensor.shape
        mean_val = np.mean(tensor)
        std_val = np.std(tensor)
        max_val = np.max(np.abs(tensor))
        
        # Visual health indicator
        health = "OK"
        if max_val < 1e-7: health = "COLLAPSE"
        if max_val > 1e10: health = "EXPLODE"
        
        # Only print in debug mode OR on critical failures
        if DEBUG_MODE:
            log_msg = (
                f"   [Neural Map] {self.name} Step {self.step_count:02} | "
                f"{layer_label:15} | Shape: {str(shape):10} | "
                f"Mean: {mean_val:8.4f} | Std: {std_val:8.4f} | {health}"
            )
            print(log_msg)
        elif health != "OK":
            print(f"   ⚠️ [{health}] {layer_label} | Shape: {shape} | Mean: {mean_val:.4f}")
        
        return tensor
