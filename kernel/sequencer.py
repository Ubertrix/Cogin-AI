import numpy as np
import gc

class SequenceGenerator:
    """
    v7.0 Creative Autoregressive Orchestrator.
    Generates deep explanations and creative text word-by-word.
    """
    def __init__(self, inference_head, tokenizer, genesis, short_term, embedding_layer, attention_layer):
        self.inference_head = inference_head
        self.tokenizer = tokenizer
        self.genesis = genesis
        self.short_term = short_term
        self.embedding_layer = embedding_layer
        self.attention_layer = attention_layer

    def generate(self, initial_vector, expert_name, input_text="", retrieved_memory_text=None, max_tokens=150, temperature=0.9, **kwargs):
        """
        Hybrid Generation: Uses retrieved knowledge as a seed for deep explanation.
        """
        # 1. If we have knowledge, use it to build a structured explanation
        if retrieved_memory_text and len(retrieved_memory_text.split()) > 10:
            explanation_intro = f"Based on my logical analysis and real-time research: "
            # Blend the knowledge with a concluding thought
            conclusion = "\n\nIn conclusion, this phenomenon demonstrates the fundamental principles of physics and logic as applied to your query."
            return f"{explanation_intro}{retrieved_memory_text}{conclusion}", 0.99

        # 2. Pure Autoregressive Generation for creative/logical tasks
        print(f"   [GENERATOR] No direct knowledge found. Generating from neural weights...")
        
        H0 = np.atleast_2d(initial_vector)
        generated_tokens = []
        current_vector = H0
        token_probs = []
        
        # Start with a logical prompt if it's an explanation
        if any(w in input_text.lower() for w in ["explain", "why", "how"]):
            intro_tokens = self.tokenizer.encode("The reason for this is")
            generated_tokens.extend(intro_tokens)
            # Update vector with intro
            intro_emb = self.embedding_layer.forward(intro_tokens)
            current_vector = np.mean(intro_emb, axis=0, keepdims=True)

        for i in range(max_tokens):
            logits = self.inference_head.forward(current_vector)
            token_id, prob = self.inference_head.sample(logits, temperature=temperature, token_history=generated_tokens)
            
            if token_id in [self.tokenizer.word2id.get("<EOS>"), self.tokenizer.word2id.get("<PAD>")]:
                if i > 20: break
                
            generated_tokens.append(token_id)
            token_probs.append(prob)
            
            # Update context vector (Autoregressive Feedback)
            token_emb = self.embedding_layer.forward([token_id])
            current_vector = 0.6 * current_vector + 0.4 * token_emb
            
            if i % 20 == 0: gc.collect()

        response_text = self.tokenizer.decode(generated_tokens)
        avg_confidence = np.mean(token_probs) if token_probs else 0.5
        
        return response_text, avg_confidence
