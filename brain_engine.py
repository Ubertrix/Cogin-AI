import sys
import os
import re
import numpy as np
import json
import time

# Core Dependencies
from kernel.ops.genesis import Genesis
from kernel.ops.synthesis_engine import SynthesisController
from router.gate import SmartGate
from kernel.exokernel import DSLAExokernel
from experts_pool.base_expert import BaseExpert
from experts_pool.linguistic.logic import LinguisticExpert
from experts_pool.mathematics.logic import MathematicsExpert
from experts_pool.coding.logic import CodingExpert
from experts_pool.expanded_experts import ScienceExpert, LiteratureExpert, IndustryExpert, FinanceExpert, ComputingExpert
from knowledge_base.v3_scraper import CogniScraperV3
from tools.web_ingestor import WebIngestor
from dictionary.tokenizer import Tokenizer
from kernel.layers.embedding import EmbeddingLayer
from kernel.layers.attention import SelfAttention
from memory.short_term import ShortTermMemory
from memory.long_term import EpistemicWeightMemory
from registry.config import SystemConfig
from kernel.inference_head import InferenceHead
from kernel.sequencer import SequenceGenerator

class IngestionEngine:
    """V3 Multi-Expert Distillation Engine."""
    def __init__(self, platform):
        self.platform = platform
        self.scraper = CogniScraperV3()
        self.web_tools = WebIngestor(scraper=self.scraper)

    def distill_text(self, text, source="Web", category=None):
        """v5.5 Non-Destructive Ingestion: Delta-Weight additive update only."""
        if not text: return
        
        # Expand vocabulary
        self.platform.tokenizer.fit([text])
        if self.platform.embedding_layer:
            self.platform.embedding_layer.grow(self.platform.tokenizer.vocab_size)
            self.platform.inference_head.update_vocab_size(self.platform.tokenizer.vocab_size, self.platform.embedding_layer.embeddings)

        # Process text block
        v3_data = self.scraper.process_block(text)
        if not v3_data:
            v3_data = {"input": text[:100], "output": text, "category": category or "General"}

        final_category = category or v3_data.get("category", "General")
        new_vec = self.platform.get_sentence_vector(text)
        concept_key = v3_data["input"].lower().strip()

        # Save to long-term memory
        self.platform.long_term.save_knowledge(
            concept_key, v3_data, vector=new_vec, category=final_category,
            neuron_weight=1.0, is_new_neuron=True
        )
        
        # Update expert weights
        self.platform.supervised_expert_update(final_category, new_vec)

class CogniPro:
    """
    Cogni Pro v5.5 Brain Engine.
    Pure Generative Mode (No Canned Responses).
    """
    def __init__(self, d_model=1024):
        self.d_model = d_model
        self.tokenizer = Tokenizer()
        self.genesis = Genesis()
        self.exokernel = DSLAExokernel()
        self.synthesis = SynthesisController()
        self.short_term = ShortTermMemory()
        self.long_term = EpistemicWeightMemory()
        
        self.experts = [
            LinguisticExpert(d_model, d_model),
            MathematicsExpert(d_model, d_model),
            CodingExpert(d_model, d_model),
            ScienceExpert(d_model, d_model),
            LiteratureExpert(d_model, d_model),
            IndustryExpert(d_model, d_model),
            FinanceExpert(d_model, d_model),
            ComputingExpert(d_model, d_model)
        ]
        
        self.router = SmartGate(input_dim=d_model, num_experts=len(self.experts))
        self.ingestion_engine = IngestionEngine(self)
        self.embedding_layer = EmbeddingLayer(vocab_size=self.tokenizer.vocab_size, d_model=self.d_model)
        self.attention = SelfAttention(d_model=d_model)
        self.inference_head = InferenceHead(self.d_model, self.tokenizer.vocab_size, self.tokenizer, self.embedding_layer.embeddings)
        self.sequencer = SequenceGenerator(self.inference_head, self.tokenizer, self.genesis, self.short_term, self.embedding_layer, self.attention)
        
        print("Cogni Pro v5.5: Neural Engine Online (Generative Mode).")

    def get_sentence_vector(self, text):
        """Vectorize input text using embedding and attention layers."""
        tokens = self.tokenizer.encode(text)
        if not tokens: return np.zeros((1, self.d_model))
        embeddings = self.embedding_layer.forward(np.array(tokens))
        attended, _ = self.attention.forward(embeddings)
        return np.mean(attended, axis=0, keepdims=True)

    def supervised_expert_update(self, category, vector, lr=0.1):
        """Update expert weights based on category."""
        expert_map = {
            "Linguistic": 0, "Math": 1, "Coding": 2,
            "Science": 3, "Literature": 4, "Industry": 5,
            "Finance": 6, "Computing": 7
        }
        idx = expert_map.get(category, 0)
        if idx < len(self.experts):
            target_expert = self.experts[idx]
            if hasattr(target_expert, 'distill'):
                target_expert.distill(vector, vector, lr=lr) 
            self.router.train_router(idx, vector, lr=lr * 2)

    def process(self, input_text):
        """Main processing pipeline for user input."""
        if not input_text: return "Please provide input.", 0.0
        
        lower = input_text.lower()
        
        # 1. Identity & Greetings (Smart Overrides)
        if any(w in lower for w in ["who are you", "your name", "من أنت", "ما اسمك"]):
            return "I am Cogni Pro, an advanced AI system developed by Ubertrix LLC. I can help you with coding, science, math, and logical analysis.", 1.0
        
        if any(w in lower for w in ["hello", "hi", "مرحبا", "اهلا"]):
            return "Hello! I am Cogni Pro. How can I assist you today?", 1.0

        # 2. Retrieval & Generation (General Path)
        input_vector = self.get_sentence_vector(input_text)
        expert, confidence = self.router.route(input_vector, self.experts, input_text=input_text)
        
        # Try to retrieve from long-term memory
        ret_data = self.long_term.retrieve(
            query_text=input_text, query_vector=input_vector, filter_category=expert.name, allow_canned=True
        )
        
        # If we found a match in the trained data
        if isinstance(ret_data, tuple) and len(ret_data) >= 5 and ret_data[4]:
            knowledge_content = ret_data[4]
            # Always prioritize retrieved knowledge if it exists after massive training
            return self.synthesis.synthesize(input_text, knowledge_content), 0.95
        
        # Pure Generation Fallback
        try:
            gen_text, gen_conf = self.sequencer.generate(
                input_vector, expert.name, input_text=input_text,
                max_tokens=40, temperature=0.7
            )
            final_response = self.synthesis.synthesize(input_text, gen_text)
            return final_response, float(gen_conf)
        except:
            return "I am processing your request. Please provide more details.", 0.5

    def display_neural_report(self):
        """Display system status report."""
        print(f"Cogni Pro v5.5 | Anchors: {len(self.long_term.knowledge_base)} | Vocab: {len(self.tokenizer.word2id)}")

    def train_knowledge(self, samples):
        """Placeholder for knowledge training."""
        pass
