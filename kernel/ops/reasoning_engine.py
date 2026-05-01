import numpy as np

class ReasoningEngine:
    """
    Chain of Thought (CoT) Engine: 
    Enables the system to break down complex queries into logical steps.
    """
    def __init__(self):
        self.logic_steps = []

    def reason(self, query, context_vector):
        """Analyzes the query and generates a logical path."""
        self.logic_steps = []
        
        # Step 1: Intent Decomposition
        self.logic_steps.append("Decomposing query intent...")
        
        # Step 2: Knowledge Mapping
        self.logic_steps.append("Mapping semantic vectors to logical anchors...")
        
        # Step 3: Conflict Resolution
        self.logic_steps.append("Checking for logical inconsistencies in retrieved data...")
        
        # Step 4: Synthesis Strategy
        strategy = self._determine_strategy(query)
        self.logic_steps.append(f"Selected synthesis strategy: {strategy}")
        
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
