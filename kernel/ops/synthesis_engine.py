import re
import json
import os
import difflib

class SynthesisController:
    """
    Inference Synthesis Engine: Transforms raw retrieval data into synthesized,
    concise, and instruction-aligned generative responses.
    Modified: Removed canned responses to allow for pure text generation.
    """
    def __init__(self, arabic_anchors_path="registry/arabic_anchors.json"):
        self.arabic_anchors = []
        self.session_history = []
        self.anchor_lock_counts = {}
        self.curiosity_index = 0
        
        # Code Syntax Templates - Kept for structural guidance if needed
        self.syntax_templates = {
            "Structure": ["def", "class", "import", "from", "async def"],
            "Action": ["print", "return", "if", "for", "while", "yield", "await"],
            "Comment": ["# [Documentation]", "# [Logic Logic]", "# [Epistemic Note]", "# [Cogni Pro Guard]"]
        }

        # We still load anchors but we won't use them to override the whole response
        if os.path.exists(arabic_anchors_path):
            try:
                with open(arabic_anchors_path, 'r', encoding='utf-8') as f:
                    self.arabic_anchors = json.load(f).get("arabic_anchors", [])
            except Exception as e:
                print(f"   [Synthesis] Error loading Arabic anchors: {e}")

    def suppress_noise(self, text):
        """Strip URLs, bracketed citations, and media credits."""
        if not text: return ""
        # Strip URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Strip citations like [1], [82]
        text = re.sub(r'\[\d+\]', '', text)
        
        # Clean up horizontal whitespace but preserve newlines
        lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in text.split('\n')]
        return "\n".join(l for l in lines if l)

    def distill_tokens(self, text, max_tokens=100):
        """Truncate and heuristically distill text to match token budget."""
        if not text: return ""
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        
        distilled = " ".join(tokens[:max_tokens])
        return distilled

    def apply_expert_overrides(self, input_text, current_response):
        """
        Modified: Removed hardcoded overrides to allow generative output.
        Only performs basic identity reinforcement if the response is empty.
        """
        if not current_response or len(current_response.strip()) < 2:
            input_lower = input_text.lower()
            identity_keys = ["who are you", "من أنت", "ما اسمك", "your name"]
            if any(key in input_lower for key in identity_keys):
                return "I am Cogni Pro, an advanced AI system. How can I help you today?"
            return "I am processing your request. Please provide more details."
            
        return current_response

    def synthesize(self, input_text, raw_response):
        """Main entry point for generative synthesis."""
        # 1. Clean the raw generative response
        clean_text = self.suppress_noise(raw_response)
        
        # 2. Apply expert overrides (now minimal)
        synthesized = self.apply_expert_overrides(input_text, clean_text)
        
        # 3. Final distillation if too long
        if len(synthesized.split()) > 150:
            synthesized = self.distill_tokens(synthesized, max_tokens=150)

        self.session_history.append(synthesized)
        return synthesized
