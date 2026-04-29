import numpy as np

class Gradients:
    """
    Backpropagation Engine
    Calculates the amount of change required in weights to reduce the error rate.
    """

    @staticmethod
    def compute_linear_grad(upstream_grad, last_input, weights):
        """
        Calculate derivatives for the Linear Layer
        upstream_grad: Error matrix coming from the next layer
        last_input: Inputs that caused this error
        """
        # Weight derivative (dW) = Inputs multiplied by incoming error
        dW = np.dot(last_input.T, upstream_grad)
        
        # Bias derivative (db) = Sum of incoming error
        db = np.sum(upstream_grad, axis=0, keepdims=True)
        
        # Input derivative (dX) to pass to the previous layer
        dX = np.dot(upstream_grad, weights.T)
        
        return dW, db, dX

    @staticmethod
    def relu_grad(z):
        """Derivative of ReLU activation function (returns 1 if positive, 0 if negative)"""
        return (z > 0).astype(float)

    @staticmethod
    def cross_entropy_loss(predictions, targets):
        """Calculate the error between model prediction and correct answer"""
        samples = predictions.shape[0]
        # Add a very small value to prevent mathematical collapse log(0)
        predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        
        loss = -np.sum(targets * np.log(predictions)) / samples
        return loss

    @staticmethod
    def compute_output_grad(predictions, targets):
        """Final error derivative at the network output (for Softmax layer)"""
        return predictions - targets
