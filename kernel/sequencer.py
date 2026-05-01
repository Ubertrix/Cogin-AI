import numpy as np
import gc

class SequenceGenerator:
    """
    v5.0 Semantic Orchestrator - Simplified for Stability
    """
    def __init__(self, inference_head, tokenizer, genesis, short_term, embedding_layer, attention_layer):
        self.inference_head = inference_head
        self.tokenizer = tokenizer
        self.genesis = genesis
        self.short_term = short_term
        self.embedding_layer = embedding_layer
        self.attention_layer = attention_layer
        self.stop_tokens = [".", "!", "?", "\n"]

    def generate(self, initial_vector, expert_name, input_text="", retrieved_memory_vector=None, max_tokens=30, top_p=0.9, top_k=40, temperature=0.7, min_length=5, **kwargs):
        """
        Simplified generation loop for stability.
        """
        H0 = np.atleast_2d(initial_vector)
        generated_tokens = []
        current_vector = H0
        
        for i in range(max_tokens):
            # 1. Get logits
            logits = self.inference_head.forward(
                current_vector, 
                anchor_vector=retrieved_memory_vector if i == 0 else None
            )
            
            # 2. Sample next token
            token_id, confidence = self.inference_head.sample(
                logits, 
                context_vector=current_vector,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                expert_name=expert_name,
                token_history=generated_tokens
            )
            
            # 3. Check for stop tokens
            token_str = self.tokenizer.decode([token_id]).strip()
            if token_id in [self.tokenizer.word2id.get("<eos>"), self.tokenizer.word2id.get("<PAD>")]:
                break
            
            generated_tokens.append(token_id)
            
            if i > min_length and token_str in self.stop_tokens:
                break
            
            # 4. Update current vector with Logical Context
            token_emb = self.embedding_layer.forward([token_id])
            # v5.5: Dynamic context update (0.6 Old / 0.4 New) for better logical flow
            current_vector = 0.6 * current_vector + 0.4 * token_emb
            
            if i % 10 == 0:
                gc.collect()

        if not generated_tokens:
            return "I am Cogni Pro, how can I help?", 0.5
            
        response_text = self.tokenizer.decode(generated_tokens)
        return response_text, 0.8
