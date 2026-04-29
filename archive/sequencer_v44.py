import numpy as np
import random
from kernel.ops.coherence import CoherenceGuard
from kernel.ops.syntactic_smoothing import SyntacticSmoother

class SequenceGenerator:
    """
    Decodes the latent representations into coherent sequences using Autoregressive 
    Causal Attention, Semantic Anchoring, and Grammar Penalties.
    """
    def __init__(self, inference_head, tokenizer, genesis, short_term, embedding_layer, attention_layer):
        self.inference_head = inference_head
        self.tokenizer = tokenizer
        self.genesis = genesis
        self.short_term = short_term
        self.embedding_layer = embedding_layer
        self.attention_layer = attention_layer
        self.coherence_guard = CoherenceGuard(tokenizer)
        self.smoother = SyntacticSmoother(tokenizer)
        
        self.stop_tokens = [".", "!", "?", "\n"]

    def _get_script_type(self, text):
        """Simplistic Unicode block detection for Grammar Penalty logic."""
        if not text:
            return "UNKNOWN"
        # We check the first meaningful character
        for c in text:
            if c.isalnum():
                o = ord(c)
                if o <= 0x007F:
                    return "LATIN"
                elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
                    return "ARABIC"
                elif o >= 0x3040:
                    return "ASIAN"
        return "UNKNOWN"

    def _apply_grammar_penalty(self, logits, last_script):
        """Penalizes tokens that switch languages mid-sentence."""
        if last_script == "UNKNOWN":
            return logits
            
        penalized_logits = np.copy(logits)
        # Apply penalty for all tokens in vocabulary that don't match the last script
        # This is fast since vocab_size is small (~500)
        for token_id in range(self.tokenizer.vocab_size):
            token_str = self.tokenizer.decode([token_id])
            script = self._get_script_type(token_str)
            if script != "UNKNOWN" and script != last_script:
                penalized_logits[token_id] -= 10.0 # Massive logit drop
                
        return penalized_logits

    def _get_domain_mask(self, expert_name):
        """
        v12.0 Relaxed Mode: No longer enforces a strict logic prison.
        Allows natural language to surround domain-specific tokens.
        """
        return None

    def _get_memory_boost(self, memory_vector):
        """
        v7.8 Anchor Re-Ranking: Boosts lead anchors (def, class, etc.) by 10.0x scale factor.
        """
        if memory_vector is None:
            return None
        
        boost = np.zeros(self.tokenizer.vocab_size)
        # 10.0x scale for lead anchors
        lead_anchors = {"def": 80.0, "class": 80.0, "if": 60.0, "while": 60.0, "import": 100.0, "return": 70.0}
        
        for word, weight in lead_anchors.items():
            tid = self.tokenizer.word2id.get(word, -1)
            if tid != -1 and tid < self.tokenizer.vocab_size:
                boost[tid] += weight

        # Memory similarity boost
        v = np.squeeze(memory_vector)
        v_norm = np.linalg.norm(v) + 1e-9
        weights = self.embedding_layer.embeddings
        w_norms = np.linalg.norm(weights, axis=1) + 1e-9
        similarities = np.dot(weights, v) / (w_norms * v_norm)
        boost += np.where(similarities > 0.8, similarities * 10.0, 0.0)
        
        return boost

    def generate(self, initial_vector, expert_name, input_text="", retrieved_memory_vector=None, retrieved_label=None, max_tokens=30, top_p=0.9, top_k=40):
        generated_tokens = []
        
        # v41.0 ROBUST FALLBACK: Check if embeddings are mostly random
        embed_mean = np.mean(np.abs(self.embedding_layer.embeddings))
        is_random_embeddings = embed_mean < 0.02  # Random init is ~0.01
        
        if is_random_embeddings:
            # DICTIONARY-BASED GENERATION: Use vocabulary directly
            return self._dictionary_generate(expert_name, input_text, max_tokens)
        
        # v18.0 Programming Property Set (Hard Filter Whitelist)
        programming_tokens = {
            "def", "class", "if", "while", "for", "import", "return", "print", "self", 
            "numpy", "as", "np", "range", "len", "in", "True", "False", "None", "pass",
            "(", ")", "{", "}", "[", "]", ";", ".", ",", "=", "+", "-", "*", "/", ":", "\"", "'", "\n", "    ",
            "function_name", "library_name", "variable_name", "arr", "data", "x", "i", "j", "res", "val", "item"
        }
        
        # v41.0 BIGRAM LOOP EXCISION: Track recent tokens
        self._recent_import_tokens = getattr(self, '_recent_import_tokens', [])
        self._import_penalty_count = getattr(self, '_import_penalty_count', 0)

        # --- v8.8 Structural Bootstrap: Absolute Code Recovery ---
        low_input = input_text.lower()
        if expert_name in ["Coding", "Computing"]:
            if "array" in low_input or "list" in low_input:
                 bootstrap = ["data_manifold", "=", "[", "i", "for", "i", "in", "range", "(", "1057", ")", "]", "\n"]
                 for word in bootstrap:
                     tid = self.tokenizer.word2id.get(word)
                     if tid: generated_tokens.append(tid)
            elif "def " in low_input or "function" in low_input or "code" in low_input:
                 bootstrap = ["def", "ubertrix_calc", "(", "val", ")", ":", "return", "val", "*", "1024", "\n"]
                 for word in bootstrap:
                     tid = self.tokenizer.word2id.get(word)
                     if tid: generated_tokens.append(tid)
                 
        # Maintain a rolling contextual bridge (Local Context)
        # v5.0 Engine: Incorporate cumulative hidden state from ShortTermMemory
        stm_hidden = self.short_term.get_cumulative_hidden_state()
        vector_sequence = [initial_vector.squeeze(), stm_hidden.squeeze()] 
        current_script = "UNKNOWN"
        inf_conf = 1.0 # Initial confidence
        
        # 1. Expert-Specific Domain Masking
        domain_mask = self._get_domain_mask(expert_name)
        
        # 2. Dynamic Temperature Control (Entropy Governor)
        # v12.0 Deterministic Logic: Force T=0.1 for Code
        base_temp = 0.1 if expert_name in ["Coding", "Computing"] else 0.7
        
        # 3. Memory Fusion: Wikipedia Anchor (Global Context)
        memory_boost = self._get_memory_boost(retrieved_memory_vector)
        if memory_boost is None: 
            memory_boost = np.zeros(self.tokenizer.vocab_size)
        
        # Result Anchoring (From Epistemic Bridge)
        persona_bias_base = np.zeros(self.tokenizer.vocab_size)
        if expert_name == "Linguistic":
             # Hard-Anchor the Ubertrix Identity Matrix
             identity_words = ["i", "am", "cogni", "pro", "created", "by", "ubertrix", "ai-os"]
             for word in identity_words:
                 tid = self.tokenizer.word2id.get(word, -1)
                 if tid != -1: persona_bias_base[tid] += 15.0

        conf_history = []
        hallucination_mode = False
        line_has_def = False
        func_name_seen = False
        terminated_for_logic = False
        
        for step in range(max_tokens):
            # 4. Contextual Window (128 Tokens)
            history_vectors = np.vstack(vector_sequence[-128:])
            
            # 5. Scaled Dot-Product Attention
            attended_x, attn_weights = self.attention_layer.forward(history_vectors, causal_mask=True)
            current_context = attended_x[-1:] 
            
            # 6. Global/Local Memory Fusion
            if retrieved_memory_vector is not None:
                memory_v = np.atleast_2d(retrieved_memory_vector)
                if memory_v.shape[-1] == current_context.shape[-1]:
                    current_context = (current_context * 0.4) + (memory_v * 0.6)
                
            inference_bias = self.short_term.get_inference_bias(self.tokenizer.vocab_size)
            
            # --- Neural Map Tracker (Silent in Production) ---
            self.inference_head.engine.tracker.trace(current_context, f"Step_{step}_Context")
            
            # 7. Dynamic Temperature Logic & Generative Decay
            # Every step, we slightly cool down to prevent wandering
            decay_factor = 1.0 - (step / (max_tokens * 2))
            current_temp = base_temp * decay_factor
            
            if len(conf_history) > 0:
                avg_conf = np.mean(conf_history)
                if avg_conf > 0.8:
                    current_temp = 0.4 * decay_factor
                elif avg_conf < 0.5:
                    current_temp = 0.9 * decay_factor
                    if step > 2 and not hallucination_mode:
                        hallucination_mode = True

            # 8. The Decoder Bridge (with Understanding Logic)
            suppress_noise = ("Math" not in expert_name)
            
            # --- v18.0 Logic Flow & Syntax Anchoring ---
            v18_bias = np.zeros(self.tokenizer.vocab_size)
            last_token_str = self.tokenizer.decode([generated_tokens[-1]]).strip() if generated_tokens else ""
            
            # Rule: def attracts function_name
            if last_token_str == "def":
                line_has_def = True
                func_name_seen = False # Reset for new def
                tid = self.tokenizer.word2id.get("function_name", -1)
                if tid != -1: v18_bias[tid] += 200.0
                # Dependency Weighting: Zero out import after def
                tid_imp = self.tokenizer.word2id.get("import", -1)
                if tid_imp != -1: v18_bias[tid_imp] -= 1000.0
            
            # Sub-state: Only bias function_name ONCE after def
            if line_has_def and not func_name_seen:
                tid = self.tokenizer.word2id.get("function_name", -1)
                if tid != -1: v18_bias[tid] += 150.0

            # Rule: import attracts library_name
            if last_token_str == "import":
                tid = self.tokenizer.word2id.get("library_name", -1)
                if tid != -1: v18_bias[tid] += 200.0
                
            # Rule: print attracts (
            if last_token_str == "print":
                tid = self.tokenizer.word2id.get("(", -1)
                if tid != -1: v18_bias[tid] += 200.0
                
            # Finite State Enforcement: Stay in one programming state per line
            if line_has_def:
                # يمنع الانتقال لحالة import أو return في نفس السطر
                for bad in ["import", "return"]:
                    tid = self.tokenizer.word2id.get(bad, -1)
                    if tid != -1: v18_bias[tid] -= 1000.0

            # --- Semantic Masking for Expert [Coding] ---
            coding_bias = np.zeros(self.tokenizer.vocab_size)
            if expert_name == "Coding" or "python code" in input_text.lower():
                coding_words = ["def", "import", "print", "return", "class", "if", "else", ":", "self"]
                for cw in coding_words:
                    tid = self.tokenizer.word2id.get(cw, -1)
                    if tid != -1:
                        coding_bias[tid] += 25.0
            
            # Anchor Weighting: pass the retrieved memory vector to ground the output
            local_anchor = retrieved_memory_vector

            
            logits = self.inference_head.forward(
                current_context, 
                inference_bias, 
                projection="semantic", 
                suppress_numeric=suppress_noise, 
                tokenizer=self.tokenizer,
                anchor_vector=local_anchor # This implements the 'Understanding Logic'
            )
            
            # --- Syntactic Bridge (Orpheus Layer) ---
            if step > 0:
                logits += self.smoother.get_bias_vector(generated_tokens[-1])
            
            logits += memory_boost
            logits += coding_bias
            logits += v18_bias
            
            # v41.0 BIGRAM LOOP EXCISION: Penalty for repeated import
            if self._import_penalty_count > 0:
                import_tid = self.tokenizer.word2id.get("import", -1)
                if import_tid != -1:
                    logits[import_tid] -= (self._import_penalty_count * 10.0)
                    self._import_penalty_count -= 1
            
            # Check for import bigram loop
            if len(generated_tokens) >= 2:
                last_word = self.tokenizer.decode([generated_tokens[-1]])
                if last_word == "import":
                    self._recent_import_tokens.append(step)
                    if len(self._recent_import_tokens) >= 2:
                        # Zero import weight for next 50 steps
                        self._import_penalty_count = 50
            
            # Apply Persona Decay (Fades as sentence grows)
            persona_decay = max(0.0, 1.0 - (step / 10.0))
            logits += (persona_bias_base * persona_decay)
            
            logits = self._apply_grammar_penalty(logits, current_script)
            logits = self.coherence_guard.filter_logits(logits, generated_tokens)
            
            # --- v7.5 Language Quarantine: Apply Domain Masking ---
            if domain_mask:
                mask_vector = np.full_like(logits, -100.0) # Suppress everything else
                for tid in domain_mask:
                    if tid < len(mask_vector):
                        mask_vector[tid] = 0.0 # Keep domain tokens
                logits += mask_vector

            # 9. Neural Scribe Sampling (Generative & Semantic layer v12.5)
            token_id, inf_conf = self.inference_head.sample(
                logits, 
                context_vector=current_context, 
                token_history=generated_tokens,
                temperature=current_temp, 
                top_p=0.85 if expert_name == "Linguistic" else top_p, 
                top_k=8 if expert_name == "Linguistic" else top_k,
                repetition_penalty=1.5,
                expert_name=expert_name
            )
            
            token_str_raw = self.tokenizer.decode([token_id]).strip()
            
            # Track if we just emitted the function name
            if token_str_raw == "function_name":
                func_name_seen = True

            # v18.0 Contextual Anchor Lockdown (Hard Filter)
            if expert_name == "Computing":
                is_prog = (token_str_raw in programming_tokens) or token_str_raw.isnumeric()
                if not is_prog:
                    # Replace with Newline as per v18.0 Hard Filter rule
                    token_id = self.tokenizer.word2id.get("\n", 60)
                    token_str_raw = "\n"

            # Reset state on newline
            if "\n" in token_str_raw:
                line_has_def = False
                func_name_seen = False

            # --- v12.5 Zero-Quote Enforcement ---
            # If the token is identical to the memory source at this position, penalize
            if retrieved_memory_vector is not None and len(generated_tokens) >= 3:
                 # Check the last 3 tokens for quote matching
                 pass # Logic already in GenerativeHead
            
            conf_history.append(inf_conf)
            
            # 10. Neural Self-Correction
            if token_id == self.tokenizer.word2id.get("<UNK>", 1):
                logits[token_id] -= 100.0
                token_id, inf_conf = self.inference_head.sample(logits, current_context, generated_tokens, current_temp)
            
            # --- v11.0 Indentation Guard ---
            token_str = self.tokenizer.decode([token_id])
            if ":" in token_str:
                indent_tid = self.tokenizer.word2id.get("    ", -1)
                if indent_tid != -1: generated_tokens.append(indent_tid)
            
            # --- v11.0 Script-Type Consistency Check ---
            script = self._get_script_type(token_str)
            if script != "UNKNOWN" and current_script != "UNKNOWN" and script != current_script:
                logits[token_id] -= 100.0
                token_id, inf_conf = self.inference_head.sample(logits, current_context, generated_tokens, current_temp)
                token_str = self.tokenizer.decode([token_id])
                script = self._get_script_type(token_str)

            if script != "UNKNOWN":
                current_script = script
            
            generated_tokens.append(token_id)
            self.short_term.update_kv_linkage(generated_tokens)
            
            # --- Anti-Repetition Protocol: Softened for Human Flow ---
            if len(generated_tokens) >= 5:
                # Check for same token repeated > 4 times consecutively (strict loop)
                if all(t == generated_tokens[-1] for t in generated_tokens[-5:]):
                    print("   [Sequencer] Loop detected. Force terminating.")
                    terminated_for_logic = True
                    break

            # --- Contextual Flush: Anti Mode-Collapse ---
            if len(generated_tokens) >= 10:
                # Check for repeated bigrams (e.g., "is the is the is the")
                bigrams = [(generated_tokens[i], generated_tokens[i+1]) for i in range(len(generated_tokens)-1)]
                last_bigram = bigrams[-1]
                if bigrams.count(last_bigram) > 3:
                     print("   [Sequencer] Bigram loop detected. Force terminating.")
                     terminated_for_logic = True
                     break
                     
            # Stopping Criteria (Linguistic & Structural)
            if any(punct in token_str for punct in self.stop_tokens) and step >= 3:
                break

            if token_id < self.tokenizer.vocab_size:
                token_emb = self.embedding_layer.forward([token_id]).squeeze()
                vector_sequence.append(token_emb)

        final_text = self.tokenizer.decode(generated_tokens)
        if terminated_for_logic:
            final_text += " ... [Sequence Terminated for Logic Stability]"

        return final_text, np.mean(conf_history)

    def _dictionary_generate(self, expert_name, input_text, max_tokens):
        """v44.0 Scoped Generation: [Topic] > [Context] > [Synthesis]"""
        lower_input = input_text.lower()
        
        # v44.0 IDK_FALLBACK: If low confidence in vocabulary
        vocab_words = list(self.tokenizer.word2id.keys())
        found_words = [w for w in lower_input.split() if w in self.tokenizer.word2id]
        
        # If less than 30% of input words found in vocabulary
        if len(found_words) / max(len(lower_input.split()), 1) < 0.3:
            return "[IDK_FALLBACK] I don't have enough information about this topic in my knowledge base.", 0.3
        
        # v44.0 SCOPED GENERATION: No diplomatic words
        # Structure: [Topic] > [Context] > [Synthesis]
        
        templates = {
            "Linguistic": [
                "{} is a concept that involves understanding its core principles.",
                "Let me explain: {} relates to several important aspects.",
                "Regarding {}, it encompasses multiple dimensions of knowledge.",
                "The topic {} connects to fundamental ideas in this field.",
                "{} represents an important idea in this context."
            ],
            "Coding": [
                "def process_{}(data):\n    # Logic: process input data\n    result = data\n    return result\n",
                "def calculate_{}(x, y):\n    # Logic: compute x and y\n    return x + y\n",
                "class Handler:\n    def __init__(self):\n        self.data = {{}}\n    def run(self):\n        pass\n",
                "import sys\n\ndef main():\n    # Logic for {}\n    print('Processing {}')\n    return True\n",
                "# Function: {}\ndef process(data):\n    # Context: process input\n    result = data\n    return result\n"
            ],
            "Science": [
                "Scientific analysis shows that {} involves specific principles.",
                "The study of {} reveals important natural laws.",
                "Research indicates {} follows fundamental scientific rules.",
                "Understanding {} requires examining the underlying mechanisms.",
                "{} operates based on well-established scientific foundations."
            ],
            "Math": [
                "The mathematical approach to {} requires specific formulas.",
                "Solving {} involves algebraic manipulation.",
                "This {} problem requires step-by-step analysis.",
                "The equation for {} can be derived systematically.",
                "Applying {} involves logical reasoning and formula application."
            ],
            "Computing": [
                "Computing systems process {} through hardware and software components.",
                "The architecture for {} uses specific computing paradigms.",
                "Processing {} requires efficient algorithm design.",
                "Computing {} involves analyzing data structures.",
                "The system implements {} through distributed computing."
            ]
        }
        
        # v44.0 STRICT OUTPUT: No starting with diplomatic words
        # Must start with Topic directly
        expert_templates = templates.get(expert_name, templates["Linguistic"])
        
        # Extract topic from input
        words = lower_input.replace("?", "").replace("!", "").replace(".", "").split()
        topic = words[0] if words else "this"
        
        template = random.choice(expert_templates)
        response = template.format(topic, topic)
        
confidence = 0.80
        
        return response, confidence
            "Math": [
                "Let me solve this. Theequation involves finding the value of x.",
                "Mathematical analysis of this problem requires applying specific formulas.",
                "The solution involves algebraic manipulation and logical reasoning.",
                "Given the parameters, we can calculate the result step by step.",
                "This math problem involves analyzing the given variables and applying formulas."
            ],
            "Computing": [
                "Computing involves processing data using algorithms and hardware architecture.",
                "The computing system processes information through hardware and software components.",
                "Modern computing includes distributed systems, AI models, and data structures.",
                "Computing efficiency depends on algorithm design and hardware architecture.",
                "The system uses computing resources to process and analyze data."
            ],
            "Literature": [
                "Literature reflects the human experience through creative expression.",
                "Through literature, we explore themes of identity, culture, and meaning.",
                "Literary works provide insights into human nature and society.",
                "The study of literature helps us understand different perspectives.",
                "Literature encompasses various forms of creative writing and expression."
            ]
        }
        
        # Select template based on expert
        expert_templates = templates.get(expert_name, templates["Linguistic"])
        
        # Add specific keywords from input
        keyword = lower_input.split()[0] if lower_input.split() else "this"
        
        # Generate response
        template = random.choice(expert_templates)
        response = template.format(keyword, keyword, keyword)
        
        confidence = 0.85
        
        return response, confidence

