import numpy as np

class Debugger:
    """وحدة مراقبة الأخطاء في الدماغ"""
    @staticmethod
    def inspect_tensor(name, tensor):
        print(f"Debug [{name}]: Shape {tensor.shape} | Max: {np.max(tensor):.4f} | Min: {np.min(tensor):.4f}")
        if np.isnan(tensor).any():
            print(f"WARNING: NaN values detected in {name}!")

    @staticmethod
    def log_event(message):
        print(f"[LOG]: {message}")
