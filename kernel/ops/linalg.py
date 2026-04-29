import numpy as np

class Linalg:
    """Core Linear Algebra Engine for COGNI_PRO"""
    @staticmethod
    def multiply(x, w):
        return np.dot(x, w)

    @staticmethod
    def add_bias(x, b):
        return x + b

    @staticmethod
    def softmax(x):
        # To maintain stability and prevent numerical explosion
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
