import numpy as np
from experts_pool.base_expert import BaseExpert
from kernel.shape_guard import shape_guard

class LinguisticExpert(BaseExpert):
    """خبير اللغويات المطور: تم تعديله ليدعم التوليد الحر بدلاً من الردود الجاهزة"""
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__(name="Linguistic", input_dim=input_dim, output_dim=output_dim)
        print(f"{self.name} Expert: Generative Mode Active.")

    @shape_guard
    def process(self, x, input_text=""):
        """
        Modified: Removed all hardcoded greetings and identity responses.
        Returns None for response to let the main engine's sequencer handle generation.
        """
        activity, _ = super().process(x)
        # We return None for the response string to signal that the sequencer should generate it
        return activity, None
