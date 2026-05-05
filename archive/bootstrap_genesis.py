import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import CogniPro
from registry.config import SystemConfig

def zero_knowledge_genesis():
    """
    Minimalist Bootstrapping:
    Initializes COGNI_PRO explicitly without ANY external knowledge databases, 
    forcing it to evolve intelligence purely through dynamic rank adaptation
    and terminal interaction loops natively.
    """
    print("--- 🌌 ZERO KNOWLEDGE GENESIS ---")
    print("Initializing Core...")
    
    # Clean the epistemic brain bounds (wiping old memory sets completely)
    brain_path = "brain_weights.npy"
    if os.path.exists(brain_path):
        os.remove(brain_path)
        print("Previous neural parameters destroyed. Wiping state...")
        
    engine = CogniPro()
    
    # 0 external files. We seed with absolutely nothing but basic system syntax parameters.
    print("Injecting Core Algebra limits...")
    seed = ["a b c d e f g h i j k l m n o p q r s t u v w x y z 1 2 3 4 5 6 7 8 9 0 . + - = / *"]
    engine.train_knowledge(seed)
    
    print("\n[Genesis] The network is empty. It has no conceptual framework.")
    print("[Genesis] It will respond with mathematical noise until weights stabilize...")
    print(f"\n--- {SystemConfig.PROJECT_NAME} (Ephemeral Seed Mode) ---")
    
    interaction_count = 0
    while True:
        try:
            interaction_count += 1
            user_input = input(f"\n[Step {interaction_count}] User: ").strip()
            
            if not user_input: continue
            if user_input.lower() == "/exit": break
            if user_input.startswith("/learn"):
                print("   Syntax: We rely on Epistemic Hebbian updates in this mode.")
                continue
                
            # Treat every input as both chat and structural embedding logic
            engine.process(user_input)
            
            # Sub-mode: Forced feedback update loop. If we want it to learn rapidly.
            if interaction_count % 3 == 0:
                print("   [Genesis Feedback] Distilling input structures natively into weights...")
                engine.manual_learn("Concept_" + str(interaction_count), user_input)
                
            if interaction_count >= 100:
                print("\n[Genesis Cap Reached] First Expert compiled natively through terminal entropy.")
                
        except KeyboardInterrupt: break
        except Exception as e: print(f"Error: {e}")

if __name__ == '__main__':
    zero_knowledge_genesis()
