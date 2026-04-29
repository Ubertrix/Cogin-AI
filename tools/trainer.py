import numpy as np

class Trainer:
    """Learning Module"""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def train_step(self, expert, input_data, target_output):
        # Simulation of the training process (weight optimization)
        current_output = expert.process(input_data)
        
        # Calculate Loss
        loss = target_output - current_output
        
        # Update weights (Stochastic Gradient Descent Sim)
        expert.weights += np.dot(input_data.T, loss) * self.lr
        
        print(f"Expert [{expert.name}] updated. Current Loss: {np.mean(np.abs(loss)):.4f}")
        return loss
