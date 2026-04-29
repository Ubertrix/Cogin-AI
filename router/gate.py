import numpy as np
import re

class SmartGate:
    """v5.0 Neural Router: High-fidelity expert routing with symbolic math priority."""
    def __init__(self, input_dim, num_experts):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_prototypes = np.random.randn(num_experts, input_dim) * 0.1
        self.w_gate = np.random.randn(input_dim, num_experts) * 0.1

    def _is_calculation_request(self, text):
        lower_text = text.lower().strip()
        numeric_expression = bool(re.search(r'\d+(?:\.\d+)?\s*[-+*/]\s*\d+(?:\.\d+)?', text))
        return lower_text.startswith("calculate") or numeric_expression

    def _classify_input_intent(self, text):
        """v44.0: Classify input into Knowledge_Search or Code_Generation"""
        lower = text.lower()
        
        # Code Generation triggers
        code_triggers = ["write code", "def ", "class ", "print(", "import ", "function", "code", "script", "program"]
        for w in code_triggers:
            if w in lower:
                return "Code_Generation"
        
        # Knowledge Search triggers
        search_triggers = ["what is", "how does", "explain", "tell me", "describe", "define", "meaning", "who is", "where is"]
        for w in search_triggers:
            if w in lower:
                return "Knowledge_Search"
        
        return "General"

    def route(self, x_vector, experts_list, input_text=""):
        """
        v5.0 Main Operating Protocol: Semantic Orchestrator
        [DOMAIN_GATE]: Route (A) Knowledge vs Route (B) Programming
        """
        lower_input = input_text.lower().strip()
        
        # v5.0 Intent Classification Logic
        intent = self._classify_input_intent(input_text)
        
        # --- PATH (A): Knowledge Search (Linguistic/Science/Math) ---
        knowledge_triggers = ["sanaa", "japan", "science", "history", "geography", "صنعاء", "اليمن", "اليابان", "علم"]
        if intent == "Knowledge_Search" or any(w in lower_input for w in knowledge_triggers):
            from experts_pool.linguistic.logic import LinguisticExpert
            from experts_pool.expanded_experts import ScienceExpert
            
            # Lock programming experts, prioritize Linguistic/Science
            for exp in experts_list:
                if isinstance(exp, LinguisticExpert):
                    return exp, 0.98
            return experts_list[0], 0.90 # Fallback to first expert (Linguistic)

        # --- PATH (B): Code Generation (NumPy-based Logic) ---
        code_triggers = ["code", "script", "cli", "python", "function", "def ", "import ", "logic"]
        if intent == "Code_Generation" or any(w in lower_input for w in code_triggers):
            from experts_pool.coding.logic import CodingExpert
            for exp in experts_list:
                if isinstance(exp, CodingExpert):
                    return exp, 0.98

        # --- PRIORITY: Arithmetic (v36.0) ---
        if self._is_calculation_request(input_text):
            from experts_pool.mathematics.logic import MathematicsExpert
            for exp in experts_list:
                if isinstance(exp, MathematicsExpert):
                    return exp, 1.0

        # ── PRIORITY 2: Geography & General Knowledge (v9.5 Unlocked) ──
        geo_triggers = ["japan", "tokyo", "east asia", "island", "geography", "اليابان", "طوكيو", "آسيا"]
        if any(w in lower_input for w in geo_triggers):
            from experts_pool.linguistic.logic import LinguisticExpert
            for exp in experts_list:
                if isinstance(exp, LinguisticExpert):
                    return exp, 1.0

        # ── PRIORITY 2: Chemistry & Science (Strict v7.0) ──
        chem_triggers = [
            "h2o", "co2", "ch4", "o2", "n2", "hcl", "nh3", "h2so4", "nacl",
            "co+h", "h2+o", "reaction", "molecule", "atom", "quantum", "physics"
        ]
        # v10.0 Fix: Define normalized to avoid 'NameError'
        normalized = lower_input.replace(" ", "")

        if any(f in normalized for f in chem_triggers) or any(s in lower_input for s in ["planck", "electron", "proton"]):
            from experts_pool.expanded_experts import ScienceExpert
            for exp in experts_list:
                if isinstance(exp, ScienceExpert):
                    return exp, 1.0

        # ── PRIORITY 3: Hard-Syntax Coding (v10.0 Python Recovery) ──────────────
        coding_triggers = [
            "def ", "class ", "print(", "import ", "lambda ", "return ",
            "if __name__", "python", "code", "script", "programming", "sql", "html", "coding",
            "اكتب كود", "برمجة", "سكربت", "برنامج"
        ]
        if any(w in lower_input for w in coding_triggers):
            from experts_pool.coding.logic import CodingExpert
            for exp in experts_list:
                if isinstance(exp, CodingExpert):
                    return exp, 1.0

        # ── PRIORITY 4: Symbolic Math & Operators (v7.0) ─────
        # Trigger on equations or math-specific operators
        math_ops = ["+", "-", "*", "/", "=", "^", "%", "=>", "sqrt", "log"]
        # v10.0 Fix: Safely use normalized here
        is_equation = any(op in normalized for op in ["=", "=>", "+", "-"]) and bool(re.search(r'\w', lower_input))
        
        has_math_words = any(w in lower_input for w in [
            "integral", "derivative", "matrix", "vector", "equation", "algebra",
            "calculus", "sin(", "cos(", "sum ", "factorial", "prime", "modulo"
        ])
        
        if is_equation or has_math_words:
            from experts_pool.mathematics.logic import MathematicsExpert
            for exp in experts_list:
                if isinstance(exp, MathematicsExpert):
                    return exp, 1.0

        # ── PRIORITY 5: Computing Terms ───────────────────────────────────────────────
        computing_triggers = [
            "computer", "llm", "transformer", "language model", "gpt", "bert",
            "algorithm", "complexity", "big o", "data structure", "operating system",
            "cpu", "gpu", "memory", "network", "distributed", "cloud", "npu", "alu"
        ]
        # v31.0: If the input is a synthesis request, route to Coding for sequencer activation
        if any(w in lower_input for w in computing_triggers) and any(w in lower_input for w in ["code", "function", "efficiency"]):
            from experts_pool.coding.logic import CodingExpert
            for exp in experts_list:
                if isinstance(exp, CodingExpert):
                    return exp, 1.0
                    
        if any(w in lower_input for w in computing_triggers):
            from experts_pool.expanded_experts import ComputingExpert
            for exp in experts_list:
                if isinstance(exp, ComputingExpert):
                    return exp, 1.0

        # ── PRIORITY 6: Industry (Restricted — No Cross-Talk) ────────────────────────
        industry_triggers = ["production", "scaling", "market", "manufacturing", "supply chain", "warehouse", "logistics"]
        if any(w in lower_input for w in industry_triggers):
            from experts_pool.expanded_experts import IndustryExpert
            for exp in experts_list:
                if isinstance(exp, IndustryExpert):
                    return exp, 1.0

        # ── PRIORITY 7: Conversational / Small Talk → Linguistic ─────────────────
        conversational = [
            "hello", "hi", "hey", "how are you", "how do you feel", "good morning",
            "good evening", "whats up", "sup", "bye", "goodbye", "see you", "what is up",
            "howdy", "yo", "greetings"
        ]
        if any(w in lower_input for w in conversational):
            from experts_pool.linguistic.logic import LinguisticExpert
            for exp in experts_list:
                if isinstance(exp, LinguisticExpert):
                    return exp, 1.0

        # ── GUARD: Short / Generic Input → Linguistic (prevent cosine misroute) ──────
        # Single words or short phrases with no domain match go to Linguistic
        word_count = len(lower_input.split())
        if word_count <= 2:
            from experts_pool.linguistic.logic import LinguisticExpert
            for exp in experts_list:
                if isinstance(exp, LinguisticExpert):
                    return exp, 0.75

        # ── FALLBACK: Cosine Similarity Semantic Routing ──────────────────────────────
        scores = np.dot(x_vector, self.expert_prototypes.T)
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        best_idx = np.argmax(probabilities)
        
        # v41.0 IMPROVED ROUTING: Check word count for better confidence
        word_count = len(lower_input.split())
        
        # For short inputs without domain match, prefer Linguistic
        if word_count <= 2:
            from experts_pool.linguistic.logic import LinguisticExpert
            for exp in experts_list:
                if isinstance(exp, LinguisticExpert):
                    return exp, 0.85
        
        return experts_list[best_idx], probabilities[0][best_idx]

    def train_router(self, target_idx, input_vector, lr=0.01):
        """تحديث مراكز الخبراء لتصبح أكثر دقة بمرور الوقت"""
        self.expert_prototypes[target_idx] += input_vector[0] * lr
        self.expert_prototypes[target_idx] /= np.linalg.norm(self.expert_prototypes[target_idx])

    def reset_centroids(self):
        """إعادة ضبط مراكز الخبراء لإزالة أي تلوث معنوي"""
        print("   [Router Audit] Resetting Expert Centroids for recalibration...")
        self.expert_prototypes = np.random.randn(self.num_experts, self.input_dim) * 0.1
