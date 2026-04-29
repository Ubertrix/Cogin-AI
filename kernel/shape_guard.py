import numpy as np
from functools import wraps
from registry.config import SystemConfig

# Global Debug Mode: Set to True to see all Forward Trace logs
DEBUG_MODE = False

def shape_guard(func):
    """
    Holographic Shape Guard: Intercepts and traces neural dimensions.
    Resolves [ValueError: shapes not aligned] by enforcing strict (1, 512) projection.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Trace input if possible
        input_info = "UNKNOWN"
        if len(args) > 1:
            if hasattr(args[1], 'shape'):
                input_info = str(args[1].shape)
            elif isinstance(args[1], str):
                input_info = f"string({len(args[1])})"

        # Execution Phase
        result = func(*args, **kwargs)
        
        # Expert process() usually returns a tuple: (activity_vector, text_response)
        if isinstance(result, tuple) and len(result) == 2:
            matrix_or_scalar, secondary_info = result
            
            # 1. Capture Original Shape
            orig_shape = "None"
            shape_corrected = False
            if hasattr(matrix_or_scalar, 'shape'):
                orig_shape = str(matrix_or_scalar.shape)
            elif np.isscalar(matrix_or_scalar):
                orig_shape = "Scalar"

            # 2. Force Strict Alignment (1, d_model)
            if np.isscalar(matrix_or_scalar) or (isinstance(matrix_or_scalar, np.ndarray) and matrix_or_scalar.size == 1):
                baseline = np.zeros((1, SystemConfig.EMBEDDING_DIM))
                baseline[0, 0] = float(np.squeeze(matrix_or_scalar))
                matrix_or_scalar = baseline
                shape_corrected = True
            else:
                matrix_or_scalar = np.atleast_2d(matrix_or_scalar)
                if matrix_or_scalar.shape[-1] != SystemConfig.EMBEDDING_DIM:
                    clean_vector = np.zeros((1, SystemConfig.EMBEDDING_DIM))
                    flat = matrix_or_scalar.flatten()
                    size = min(len(flat), SystemConfig.EMBEDDING_DIM)
                    clean_vector[0, :size] = flat[:size]
                    matrix_or_scalar = clean_vector
                    shape_corrected = True
                
                if matrix_or_scalar.shape[0] > 1:
                    matrix_or_scalar = matrix_or_scalar[-1:]

            # 3. Print Trace Report (only in debug mode or on corrections)
            if DEBUG_MODE or shape_corrected:
                print(f"   [Forward Trace] {func.__name__:15} | IN: {input_info:10} | OUT: {orig_shape:8} -> {str(matrix_or_scalar.shape)}")
            
            return matrix_or_scalar, secondary_info

        return result

    return wrapper
