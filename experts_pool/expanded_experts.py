from experts_pool.base_expert import BaseExpert
import re

class ScienceExpert(BaseExpert):
    """خبير العلوم: الفيزياء، الكيمياء، والبيولوجيا (v7.0 Grounding)"""
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__(name="Science", input_dim=input_dim, output_dim=output_dim)
        self.chem_data = {
            "h2o": "Distilled Logic: Water (H2O) is a polar inorganic compound. Universal Solvent / Liquid State.",
            "co2": "Distilled Logic: Carbon dioxide (CO2) is a colorless gas with a density about 53% higher than dry air.",
            "ch4": "Distilled Logic: Methane (CH4) is the simplest alkane, the main constituent of natural gas.",
            "o2":  "Distilled Logic: Oxygen (O2) is a highly reactive nonmetal and an oxidizing agent.",
            "hcl": "Distilled Logic: Hydrogen chloride (HCl) forms hydrochloric acid fumes on contact with water vapor."
        }

    def process(self, x, input_text=""):
        activity, _ = super().process(x)
        text = input_text.lower().replace(" ", "").replace("0", "o")
        
        # --- v7.0 Chemical Reaction Grounding ---
        if "co+h" in text:
             return activity, (
                 "### Chemical Synthesis: Formyl Radical (HCO)\n"
                 "Reaction: Carbon Monoxide (CO) + Hydrogen (H) → HCO\n"
                 "Logic: Fundamental step in interstellar chemistry. HCO is a key intermediate for Formaldehyde."
             )
             
        if "h2+o" in text or "2h+o" in text:
             return activity, "Reaction: 2H + O → H2O (Water Formation). State: Liquid Grounding."

        import re
        for formula, info in self.chem_data.items():
            if re.search(r'\b' + formula + r'\b', input_text.lower()):
                return activity, info
                
        if any(w in text for w in ["quantum", "planck", "electron", "proton"]):
            return activity, "Quantum Logic: Planck's constant (h) defines the discrete nature of energy levels in the 1024-D manifold."
            
        return activity, "Science Expert: Logic space active for Physical and Chemical analysis (v7.0)."


class LiteratureExpert(BaseExpert):
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__(name="Literature", input_dim=input_dim, output_dim=output_dim)

    def process(self, x, input_text=""):
        activity, _ = super().process(x)
        return activity, "Literature analysis active. Cogni Pro can assist with text comprehension, narrative structure, and linguistic patterns."


class IndustryExpert(BaseExpert):
    """v5.0 Industry Expert: Only activates for Production, Scaling, Market terms."""
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__(name="Industry", input_dim=input_dim, output_dim=output_dim)
        self.domain_responses = {
            "production":    "Production logic: Optimizing throughput via lean manufacturing and process automation principles.",
            "scaling":       "Scaling strategy: Horizontal scaling distributes load; vertical scaling increases resource capacity per node.",
            "market":        "Market analysis: Demand signals, competitive positioning, and supply chain optimization are key Industry vectors.",
            "manufacturing": "Manufacturing logic: JIT (Just-in-Time) production minimizes inventory overhead and maximizes throughput.",
            "logistics":     "Logistics analysis: Route optimization and warehouse management are core to Supply Chain efficiency.",
            "supply chain":  "Supply Chain logic: End-to-end visibility, from raw material sourcing to last-mile delivery, is the Industry standard.",
            "warehouse":     "Warehouse management: Inventory systems, SKU tracking, and pick-and-pack optimization are key efficiency levers.",
        }

    def process(self, x, input_text=""):
        activity, _ = super().process(x)
        lower = input_text.lower()
        for key, response in self.domain_responses.items():
            if key in lower:
                return activity, response
        return activity, "Industry shard active. Specify: production, scaling, market, logistics, or manufacturing context."


class FinanceExpert(BaseExpert):
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__(name="Finance", input_dim=input_dim, output_dim=output_dim)

    def process(self, x, input_text=""):
        activity, _ = super().process(x)
        lower = input_text.lower()
        if any(w in lower for w in ["revenue", "profit", "loss", "budget", "investment", "roi"]):
            return activity, "Finance logic: Capital allocation, ROI analysis, and risk-adjusted returns are foundational finance concepts."
        return activity, "Finance shard active. Areas: capital allocation, risk modeling, and market instruments analysis."


class ComputingExpert(BaseExpert):
    """v5.0 Computing Expert: Covers hardware, systems, LLMs, and distributed computing."""
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__(name="Computing", input_dim=input_dim, output_dim=output_dim)

    def process(self, x, input_text=""):
        activity, _ = super().process(x)
        lower = input_text.lower()
        if "computer" in lower:
            return activity, "Computing logic: A computer is an electronic device that processes data via its CPU, memory, and I/O systems."
        if any(w in lower for w in ["llm", "transformer", "language model", "gpt", "bert"]):
            return activity, "LLM logic: Large Language Models use Transformer architectures with self-attention to model language at scale."
        if any(w in lower for w in ["algorithm", "complexity", "big o", "data structure"]):
            return activity, "Algorithm logic: Computational complexity (Big-O notation) defines the efficiency of algorithms over input size."
        return activity, "Computing shard active. Areas: algorithms, hardware architecture, distributed systems, and AI models."
