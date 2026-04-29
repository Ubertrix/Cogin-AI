import os
import numpy as np

class EpistemicWeightMemory:
    """
    Weight-Based Knowledge Store: Replaces static JSON loops.
    Stores and retrieves purely semantic tensors (the 'Master Vector') 
    distilled directly from Neural interactions.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            # Use absolute path relative to the project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, "brain_weights.npy")
        else:
            self.db_path = db_path
        self.knowledge_base = self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.db_path):
            try:
                # Allow pickle for dictionary of tensors
                data = np.load(self.db_path, allow_pickle=True).item()
                print(f"Epistemic Weights Loaded: {len(data)} geometric anchors active.")
                return data
            except Exception as e:
                print(f"Missing Brain Weights: {e}")
                return {}
        return {}


    def _quantize(self, vector):
        """The 120B Rule: Compress 32-bit floats into 8-bit integers."""
        v = np.squeeze(vector)
        low, high = v.min(), v.max()
        if high == low:
            return np.zeros_like(v, dtype=np.uint8), float(low), float(high)
        
        q = ((v - low) / (high - low + 1e-9) * 255).astype(np.uint8)
        return q, float(low), float(high)

    def _dequantize(self, q, low, high):
        """Restore semantic precision and upscale if necessary (Epistemic Upscaler)."""
        vec = (q.astype(np.float32) / 255.0) * (high - low) + low
        # Epistemic Upscaler: Support legacy 512-D anchors in the 1024-D manifold
        if vec.shape[0] == 512:
             # Zero-padding to reach target dimension (1024)
             upscaled = np.zeros(1024, dtype=np.float32)
             upscaled[:512] = vec
             return upscaled
        return vec

    def save_knowledge(self, concept, info, vector=None, category="General", label="", **kwargs):
        """
        Information Distillation into 1024-D space.
        v4.0: Non-Destructive Hebbian Learning (Cumulative Expansion)
        """
        if vector is None:
            return
            
        concept_key = concept.lower()
        vector = np.squeeze(vector)
        
        # Ensure native 1024-D storage for new entries
        if vector.shape[0] == 512:
             upscaled = np.zeros(1024, dtype=np.float32)
             upscaled[:512] = vector
             vector = upscaled

        # v4.0 Shard Locking Logic: Check if the existing concept is foundational
        if concept_key in self.knowledge_base:
            existing = self.knowledge_base[concept_key]
            existing_label = existing.get("label", "").lower()
            existing_cat = existing.get("category", "").lower()
            # Lock foundational logic from decaying
            if any(lock in existing_label or lock in existing_cat for lock in ["quantum", "planck", "math", "identity"]):
                print(f"   [Shard Lock] Foundational anchor '{concept_key}' is locked. Skipping update.")
                return

            # v4.5 Hebbian Merging strategy (0.8 Old / 0.2 New)
            if "essence" in existing:
                old_vec = self._dequantize(existing["essence"], *existing.get("bounds", (0.0, 1.0)))
                # v11.0 Senior Architect Refinement: 0.95 Old / 0.05 New for extreme stability
                ratio_new = 0.05
                ratio_old = 0.95
                
                print(f"   [Hebbian Loop v11.0] Concept '{concept_key}' merging Synapses ({ratio_old}/{ratio_new}).")
                vector = (old_vec * ratio_old) + (vector * ratio_new)

        q_vec, low, high = self._quantize(vector)
        
        # Binary Epistemic Ingestion
        self.knowledge_base[concept_key] = {
            "essence": q_vec,
            "bounds": (low, high),
            "category": category,
            "label": label,
            "data": info,
            "block_id": f"blk_{len(self.knowledge_base)}",
            "neuron_weight": kwargs.get("neuron_weight", 1.0),
            "is_new_neuron": kwargs.get("is_new_neuron", True),
            "v4_sync": True # Cumulative Tag
        }
        
        # Binary `.npy` memory serialization.
        np.save(self.db_path, self.knowledge_base)

    def retrieve(self, query_text=None, query_vector=None, top_k=1, filter_category=None, allow_canned=False):
        """
        v5.0 [SEARCH_FIRST]: Scan 1167 anchors. 
        [IDK_FALLBACK]: If confidence < 80%, request context.
        """
        if not self.knowledge_base or query_vector is None:
            return ("[IDK_FALLBACK]", np.zeros(1024), "Unknown", "No match", "No knowledge base available."), "Context needed"
        
        q_vec = np.squeeze(query_vector)
        q_norm = np.linalg.norm(q_vec) + 1e-9
        q_text = query_text.lower() if query_text else ""
        
        # Check vocab coverage for anchors
        found_words = [w for w in q_text.split() if w in self.knowledge_base]
        vocab_coverage = len(found_words) / max(len(q_text.split()), 1)
        
        results = []
        for concept, bundle in self.knowledge_base.items():
            category = bundle.get("category", "General")
            if filter_category and filter_category.lower() not in category.lower():
                continue

            db_vec = self._dequantize(bundle["essence"], *bundle.get("bounds", (0.0, 1.0)))
            db_norm = np.linalg.norm(db_vec) + 1e-9
            
            # Semantic similarity (Cosine)
            sim = np.dot(q_vec, db_vec) / (q_norm * db_norm)
            
            label = bundle.get("label", "")
            data_text = bundle.get("data", label)
            results.append((sim, concept, np.atleast_2d(db_vec), category, label, data_text))

        results.sort(key=lambda x: x[0], reverse=True)
        
        if top_k == 1:
            # v5.0 [SEARCH_FIRST] Threshold: 80% (0.80)
            if results:
                res = results[0]
                sim = res[0]
                # Only return canned/pre-saved text when explicitly allowed
                if allow_canned and (sim > 0.80):
                    return res[1], res[2], res[3], res[4], res[5]
                # Otherwise return the vector and metadata but not the stored text
                return res[1], res[2], res[3], res[4], None

            # v5.0 [IDK_FALLBACK]: Ask for more context instead of Gothic/Vague links
            fallback_text = "I need more context to provide an accurate answer. Could you clarify your request?"
            return ("[IDK_FALLBACK]", np.zeros(1024), "Unknown", "Low confidence", fallback_text)
            
        return results[:top_k]
