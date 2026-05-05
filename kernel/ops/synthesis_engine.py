import re
import json
import os

class SynthesisController:
    """
    v6.0 Neural Synthesis Controller.
    Refines and structures the output of the Autoregressive Generator.
    """
    def __init__(self, arabic_anchors_path="registry/arabic_anchors.json"):
        self.session_history = []
        
    def suppress_noise(self, text):
        """Strip URLs, bracketed citations, and special tokens."""
        if not text: return ""
        # Strip URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Strip citations like [1], [82]
        text = re.sub(r'\[\d+\]', '', text)
        # Strip special tokens
        text = text.replace("<PAD>", "").replace("<UNK>", "").replace("<SOS>", "").replace("<EOS>", "")
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def synthesize(self, input_text, raw_response):
        """Main entry point for generative synthesis."""
        # 1. Clean the raw generative response
        clean_text = self.suppress_noise(raw_response)
        
        # 2. Identity reinforcement if response is too short or noisy
        if len(clean_text.split()) < 3:
            input_lower = input_text.lower()
            if any(key in input_lower for key in ["who are you", "من أنت", "ما اسمك"]):
                clean_text = "I am Cogni Pro, an advanced AI system designed for deep reasoning and generation."
            elif any(key in input_lower for key in ["hello", "hi", "مرحبا"]):
                clean_text = "Hello! I am Cogni Pro. How can I assist you today?"
        
        # 3. Structural Polish
        if clean_text and not clean_text[0].isupper() and clean_text[0].isalpha():
            clean_text = clean_text[0].upper() + clean_text[1:]
            
        if clean_text and clean_text[-1] not in [".", "!", "?"]:
            clean_text += "."

        self.session_history.append(clean_text)
        return clean_text
