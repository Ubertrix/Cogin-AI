import os
import json

class CodeLearner:
    """
    v1.0 Code Self-Learning Module.
    Extracts patterns and logic from code to expand Cogni Pro's coding knowledge.
    """
    def __init__(self, brain_engine):
        self.brain = brain_engine

    def learn_from_code(self, filename, content, language):
        """
        Processes code content and integrates it into the brain's memory.
        """
        print(f"   [LEARNING] Analyzing {language} code from {filename}...")
        
        # 1. Extract key functions or classes (Simple Regex for now)
        # In a real scenario, we'd use an AST parser
        patterns = []
        if language == "Python":
            patterns = self._extract_python_patterns(content)
        elif language == "JavaScript":
            patterns = self._extract_js_patterns(content)
            
        # 2. Distill into knowledge base
        for p in patterns:
            knowledge_entry = {
                "input": f"How to implement {p['name']} in {language}?",
                "output": p['code'],
                "category": "Coding",
                "language": language,
                "type": p['type']
            }
            
            # Save to brain's long term memory
            concept_key = f"code_{language}_{p['name']}".lower()
            vector = self.brain.get_sentence_vector(knowledge_entry["input"])
            
            self.brain.long_term.save_knowledge(
                concept_key, 
                knowledge_entry, 
                vector=vector, 
                category="Coding"
            )
            
        print(f"   [LEARNING] Successfully learned {len(patterns)} patterns from {filename}.")

    def _extract_python_patterns(self, content):
        patterns = []
        # Extract functions
        funcs = [] # Simple logic to find defs
        lines = content.split('\n')
        current_func = None
        current_code = []
        
        for line in lines:
            if line.strip().startswith("def "):
                if current_func:
                    patterns.append({"name": current_func, "code": "\n".join(current_code), "type": "function"})
                current_func = line.split('(')[0].replace('def ', '').strip()
                current_code = [line]
            elif current_func and (line.startswith(' ') or line.startswith('\t') or not line.strip()):
                current_code.append(line)
            elif current_func:
                patterns.append({"name": current_func, "code": "\n".join(current_code), "type": "function"})
                current_func = None
                current_code = []
        
        return patterns

    def _extract_js_patterns(self, content):
        # Similar logic for JS
        return []
