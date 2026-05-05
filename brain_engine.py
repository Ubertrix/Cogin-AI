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
from tools.speech_engine import SpeechEngine
from tools.live_search import LiveSearchTool
from memory.dynamic_storage import DynamicStorageModule
from tools.github_fetcher import GitHubFetcher
from tools.code_learner import CodeLearner
import re
from dictionary.tokenizer import Tokenizer
from kernel.layers.embedding import EmbeddingLayer
from kernel.layers.attention import SelfAttention
from memory.short_term import ShortTermMemory
from memory.long_term import EpistemicWeightMemory
from registry.config import SystemConfig
from kernel.inference_head import InferenceHead
from kernel.sequencer import SequenceGenerator
from kernel.ops.reasoning_engine import ReasoningEngine

class IngestionEngine:
    """V3 Multi-Expert Distillation Engine."""
    def __init__(self, platform):
        self.platform = platform
        self.scraper = CogniScraperV3()
        self.web_tools = WebIngestor(scraper=self.scraper)

    def distill_text(self, text, source="Web", category=None):
        """v6.0 Real-time Knowledge Distillation."""
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
    Cogni Pro v6.0 Brain Engine.
    True Autoregressive Generation & Self-Learning.
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
        self.reasoner = ReasoningEngine()
        self.speech = SpeechEngine()
        self.live_search = LiveSearchTool()
        self.dynamic_storage = DynamicStorageModule()
        self.github_fetcher = GitHubFetcher()
        self.code_learner = CodeLearner(self)

    def get_sentence_vector(self, text):
        tokens = self.tokenizer.encode(text)
        if not tokens: return np.zeros(self.d_model)
        embs = self.embedding_layer.forward(tokens)
        return np.mean(embs, axis=0)

    def supervised_expert_update(self, category, vector):
        for expert in self.experts:
            if expert.name.lower() == category.lower():
                # Use distill for weight updates
                expert.distill(vector, vector)

    def process(self, input_text):
        """
        Main processing loop: Reasoning -> Retrieval/Search -> Learning -> Generation.
        """
        # 1. Vectorize Input
        input_vector = self.get_sentence_vector(input_text)

        # 1.5 Check for GitHub Links
        github_match = re.search(r'github\.com/([\w-]+/[\w.-]+)', input_text)
        if github_match:
            repo_path = github_match.group(1)
            # Try to extract file path if present
            file_match = re.search(r'blob/[\w-]+/(.+)', input_text)
            file_path = file_match.group(1) if file_match else ""
            
            repo_data = self.github_fetcher.fetch_repo_code(repo_path, file_path)
            if repo_data:
                for item in repo_data:
                    lang = self.github_fetcher.detect_language(item['name'], item['content'])
                    self.code_learner.learn_from_code(item['name'], item['content'], lang)
                return f"I have successfully accessed the GitHub repository '{repo_path}', analyzed the code, and integrated the patterns into my knowledge base. I am now ready to help you with this codebase.", 1.0
        
        # 2. Expert Routing
        expert, confidence = self.router.route(input_vector, self.experts, input_text=input_text)
        
        # 3. Reasoning Phase (Chain of Thought)
        self.reasoner.reason(input_text, input_vector)
        
        # 4. Memory Retrieval & Search Logic
        knowledge_content = None
        
        # Try to find code patterns first if it's a coding query
        if expert.name == "Coding":
            code_ret = self.long_term.retrieve(
                query_text=input_text, query_vector=input_vector, filter_category="Coding"
            )
            if code_ret and isinstance(code_ret, tuple) and len(code_ret) >= 5 and code_ret[4]:
                knowledge_content = code_ret[4]
                print(f"   [MEMORY] Code pattern found in local memory.")

        ret_data = None
        if not knowledge_content:
            ret_data = self.long_term.retrieve(
                query_text=input_text, query_vector=input_vector, filter_category=expert.name
            )
        
            if isinstance(ret_data, tuple) and len(ret_data) >= 5 and ret_data[4]:
                # Ensure it's not a canned response
                if "I need more context" not in ret_data[4]:
                    knowledge_content = ret_data[4]
                    print(f"   [MEMORY] Knowledge found in local memory.")
        
        # 5. Trigger Live Search if needed
        if not knowledge_content:
            print(f"   [MEMORY] No local knowledge. Triggering Live Search...")
            search_result = self.live_search.search(input_text)
            if search_result:
                print(f"   [SELF_LEARNING] Learning new information...")
                self.ingestion_engine.distill_text(search_result, source="LiveSearch")
                self.dynamic_storage.save_knowledge(input_text, search_result, input_vector)
                knowledge_content = search_result
            else:
                print(f"   [SEARCH] No live information found. Falling back to pure generation.")

        # 6. Hybrid Autoregressive Generation
        try:
            gen_text, gen_conf = self.sequencer.generate(
                input_vector, expert.name, 
                input_text=input_text,
                retrieved_memory_text=knowledge_content,
                max_tokens=100, temperature=0.8
            )
            
            # Final Synthesis
            final_response = self.synthesis.synthesize(input_text, gen_text)
            
            self.speech.speak(final_response)
            return final_response, float(gen_conf)
        except Exception as e:
            print(f"   [Generation Error] {e}")
            return "I am Cogni Pro, an AI system. I am currently processing your request.", 0.5

    def display_neural_report(self):
        print(f"Cogni Pro v6.0 | Anchors: {len(self.long_term.knowledge_base)} | Vocab: {len(self.tokenizer.word2id)}")
