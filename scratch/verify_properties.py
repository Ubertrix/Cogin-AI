from main import CogniPro
import numpy as np

# Mocking enough to run process
cp = CogniPro()

print("--- Property Seeking Test ---")
input_text = "H2o"
response, conf = cp.process(input_text)
print(f"Input: {input_text} -> Response: {response}")
