from kernel.ops.synthesis_engine import SynthesisController
from main import CogniPro
import numpy as np

sc = SynthesisController()

print("--- Fuzzy Contextual Bridging (Short Patterns) ---")
# bf should be unrecognized or fuzzy matched
print(f"Input: bf -> {sc.synthesize('bf', 'raw')}")
print(f"Input: googl -> {sc.synthesize('googl', 'raw')}")

print("\n--- Anchor Diversity Lock (3-Use Limit) ---")
sc.arabic_anchors = [{"text": "فلسفة العلم"}]
query = "فلسفة"
for i in range(5):
    print(f"Attempt {i+1}: {sc.synthesize(query, 'raw')}")

print("\n--- Code Syntax Injection (Structure -> Action -> Comment) ---")
print(sc.synthesize("code", "class ..."))

print("\n--- Multi-Step Verification (First Token Check) ---")
cp = CogniPro()
cp.prev_first_token = cp.tokenizer.encode("The")[0]
# We mock a response that would have started with "The"
print("Triggering re-sample check (Mock prev_first_token = 'The')")
resp, conf = cp.process("Hello world") 
print(f"Response: {resp}")
# If successful, cp.prev_first_token will be updated.
