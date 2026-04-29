import numpy as np
from kernel.sequencer import SequenceGenerator

class MockTokenizer:
    vocab_size = 100
    word2id = {"self": 10, "return": 11}
    def decode(self, ids):
        if ids == [10]: return "self"
        if ids == [11]: return "return"
        return "unknown"

class MockInferenceHead:
    def sample(self, *args, **kwargs):
        return 10, 0.9 # Alway return 'self'
    def forward(self, *args, **kwargs):
        return np.zeros(100)

class MockShortTerm:
    history = []
    def get_cumulative_hidden_state(self): return np.zeros((1, 512))
    def get_inference_bias(self, size): return np.zeros(size)
    def update_kv_linkage(self, tokens): pass

class MockEmbedding:
    def forward(self, ids): return np.zeros((1, 512))

class MockAttention:
    def forward(self, x, **kwargs): return x, None

sg = SequenceGenerator(
    MockInferenceHead(),
    MockTokenizer(),
    None, # genesis
    MockShortTerm(),
    MockEmbedding(),
    MockAttention()
)

print("--- Repetition Test ---")
response, conf = sg.generate(np.zeros((1, 512)), "Coding")
print(f"Response: {response}")
