import numpy as np

class BrainTeacher:
    """Supervised Learning Module"""
    def __init__(self, target_experts_count):
        self.target_experts_count = target_experts_count
        self.correction_factor = 0.05

    def supervise(self, input_vector, router_weights, labels):
        """
        This module compares the router's choice with the correct choice (Labels)
        and adjusts the router's weights to reduce error in future iterations.
        """
        # Simulation of the adjustment process
        target_expert_idx = labels[0]
        
        # Optimize gate weights (Simulation)
        router_weights[:, target_expert_idx] += self.correction_factor
        
        print(f"Teacher: Routing logic adjusted for Expert #{target_expert_idx}")
        return router_weights
