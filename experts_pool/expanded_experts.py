from experts_pool.base_expert import BaseExpert

class ScienceExpert(BaseExpert):
    def __init__(self, d_model=1024, d_ff=4096):
        super().__init__("Science")

class LiteratureExpert(BaseExpert):
    def __init__(self, d_model=1024, d_ff=4096):
        super().__init__("Literature")

class IndustryExpert(BaseExpert):
    def __init__(self, d_model=1024, d_ff=4096):
        super().__init__("Industry")

class FinanceExpert(BaseExpert):
    def __init__(self, d_model=1024, d_ff=4096):
        super().__init__("Finance")

class ComputingExpert(BaseExpert):
    def __init__(self, d_model=1024, d_ff=4096):
        super().__init__("Computing")
