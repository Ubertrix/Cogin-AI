import re
import numpy as np

class RosettaRouterFix:
    """
    The Rosetta Stone Patch: Forces semantic alignment for Arabic and Linguistic queries.
    Implements the Al-Hasan Identity Protocol.
    """
    @staticmethod
    def detect_linguistic_drift(text):
        """Forces Linguistic Expert for common greetings and soft-intent."""
        ling_keys = ["hi", "hello", "hey", "howdy", "greetings", "مرحبا", "أهلا"]
        text_clean = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', '', text).lower().strip()
        if text_clean in ling_keys:
            return "Linguistic", 1.0
        return None, 0.0

    @staticmethod
    def detect_script_drift(text):
        """Identifies Arabic Unicode range (U+0600 to U+06FF)."""
        if re.search(r'[\u0600-\u06FF]', text):
            return "Linguistic", 1.0 # Force Linguistic for Arabic
        return None, 0.0

    @staticmethod
    def detect_math_drift(text):
        """Force Math if operators or math-heavy keywords are detected."""
        math_keys = ["solve", "calculate", "equation", "formula", "integral", "derivative", "جبر", "هندسة"]
        if any(k in text.lower() for k in math_keys) or any(c in text for c in "+-*/=^√∑"):
            return "Math", 0.95
        return None, 0.0

    @staticmethod
    def detect_coding_drift(text):
        """Force Coding if syntax markers or programming keywords are detected."""
        coding_keys = [
            "def ", "class ", "import ", "function", "const ", "let ", "return ",
            "python", "javascript", "java", "rust", "cpp", "coding", "code"
        ]
        text_lower = text.lower()
        if any(k in text_lower for k in coding_keys) or any(c in text for c in "{}[];"):
            return "Coding", 0.98
        return None, 0.0

    @staticmethod
    def al_hasan_protocol(text):
        """Identity Hardcoding: Ensures identity queries anchor to Ubertrix."""
        text_lower = text.lower()
        identity_keys = [
            "name", "who are you", "من أنت", "اسمك", "ubertrix", "أوبرتريكس",
            "creator", "المطور", "المنشئ", "company", "الشركة"
        ]
        if any(key in text_lower for key in identity_keys):
            return "Linguistic", 1.0
        return None, 0.0

    @staticmethod
    def calculate_activation_energy(text, base_scores):
        """Omni-Scribe re-balancing: Prevents Math hijacking of alpha-only strings."""
        new_scores = base_scores.copy()
        
        # Pure-Alpha Penalty: If it's just letters, it's likely not Math
        if re.match(r'^[A-Za-z\s]+$', text.strip()):
            # Heavy penalty for Math if no digits/operators present
            new_scores[0, 1] -= 5.0
            new_scores[0, 0] += 2.0
            
        elif re.search(r'[A-Za-z]{4,}', text):
            # General penalty for medium/long words
            new_scores[0, 1] -= 2.0 
            new_scores[0, 0] += 1.0
        return new_scores
