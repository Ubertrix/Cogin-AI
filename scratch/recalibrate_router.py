from main import CogniPro
import numpy as np

cp = CogniPro()

print("--- System Recalibration (Router Expansion) ---")
# The router prototypes need to be pushed toward their respective semantic clusters
training_samples = [
    (0, "hello hi greeting linguistic language rosetta translation"), 
    (1, "calculate math algebra numbers equation derivative integral"), 
    (2, "python code programming logic software development function"),
    (3, "physics science biology molecule quantum chemistry laboratory"),
    (4, "poetry novel literature history philosophy epistemic wisdom"),
    (5, "factory industry manufacturing logistics production automation"),
    (6, "finance market stock investment revenue trading banking fiscal"),
    (7, "computing cpu gpu linux operating system hardware motherboard")
]

for idx, sample in training_samples:
    print(f"Calibrating Expert {idx} ({cp.experts[idx].name})...")
    # Run multiple times to strengthen the centroid positioning
    for _ in range(10):
        vector = cp.get_sentence_vector(sample)
        cp.router.train_router(idx, vector, lr=0.5)

# Verify Calibration
print("\n--- Calibration Check ---")
for idx, sample in training_samples:
    vector = cp.get_sentence_vector(sample)
    scores = np.dot(vector, cp.router.expert_prototypes.T)
    best_idx = np.argmax(scores)
    print(f"Sample: '{sample[:20]}...' -> Routed to index: {best_idx} (Target: {idx})")

print("\nRecalibration Complete.")
