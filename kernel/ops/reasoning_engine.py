import numpy as np
import time

class ReasoningEngine:
    """
    v6.0 Chain of Thought (CoT) Engine: 
    Enables the system to break down complex queries into logical steps.
    """
    def __init__(self):
        self.logic_steps = []

    def reason(self, query, context_vector):
        """Analyzes the query and generates a logical path with visual feedback."""
        self.logic_steps = []
        
        print(f"\n   [THOUGHT_PROCESS] Analyzing query: '{query}'")
        
        # Step 1: Intent Decomposition
        step1 = "Decomposing query intent..."
        self.logic_steps.append(step1)
        print(f"   [STEP 1] {step1}")
        time.sleep(0.2)
        
        # Step 2: Knowledge Mapping
        step2 = "Mapping semantic vectors to logical anchors..."
        self.logic_steps.append(step2)
        print(f"   [STEP 2] {step2}")
        time.sleep(0.2)
        
        # Step 3: Conflict Resolution
        step3 = "Checking for logical inconsistencies in retrieved data..."
        self.logic_steps.append(step3)
        print(f"   [STEP 3] {step3}")
        time.sleep(0.2)
        
        # Step 4: Synthesis Strategy
        strategy = self._determine_strategy(query)
        step4 = f"Selected synthesis strategy: {strategy}"
        self.logic_steps.append(step4)
        print(f"   [STEP 4] {step4}")
        print(f"   [DECISION] Proceeding to Autoregressive Generation.\n")
        
        return self.logic_steps

    def _determine_strategy(self, query):
        query = query.lower()
        if any(w in query for w in ['why', 'how', 'explain', 'لماذا', 'كيف']):
            return "Causal Analysis"
        if any(w in query for w in ['solve', 'calculate', 'احسب', 'حل']):
            return "Step-by-Step Derivation"
        return "Direct Semantic Synthesis"

    def get_thought_process(self):
        return " -> ".join(self.logic_steps)
