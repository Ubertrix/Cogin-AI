import numpy as np

class SyntacticSmoother:
    """
    The Orpheus Bridge: Provides syntactic probability anchors for conversational logic.
    Biases the model towards human-like token sequences (Grammar Glue).
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Transition Matrix (Conceptual Bigrams)
        # {current_word: {next_word: boost_value}}
        self.transitions = {
            "i": {"am": 20.0, "can": 15.0, "will": 15.0, "have": 10.0, "m": 25.0},
            "am": {"a": 10.0, "cogni": 20.0, "pro": 15.0, "an": 10.0, "created": 15.0},
            "how": {"are": 25.0, "can": 15.0},
            "are": {"you": 25.0, "doing": 10.0},
            "you": {"are": 15.0, "?": 10.0, "can": 10.0},
            "what": {"is": 25.0, "are": 15.0, "can": 10.0},
            "is": {"a": 10.0, "the": 10.0, "an": 10.0, "my": 10.0},
            "my": {"name": 20.0, "creator": 15.0},
            "hello": {",": 10.0, "world": 10.0, "!": 10.0, "i": 15.0},
            "مرحبا": {"بك": 20.0, "يا": 15.0},
            "أنا": {"نظام": 20.0, "ذكاء": 15.0, "كوجني": 20.0},
            "نظام": {"كوجني": 20.0, "ذكي": 15.0},
            "كوجني": {"برو": 30.0},
            "برو": {"هو": 15.0},
            "كيف": {"حالك": 20.0, "يمكنني": 20.0},
            "يمكنني": {"مساعدتك": 25.0},
            "من": {"أنت": 25.0},
            "أنت": {"نظام": 15.0},
            # Programming Transitions (v18.0)
            "def": {"function_name": 100.0, "(": 50.0, "import": -500.0, "return": -500.0},
            "import": {"library_name": 80.0, "os": 40.0, "sys": 40.0, "json": 40.0},
            "print": {"(": 100.0}
        }

    def get_bias_vector(self, last_token_id):
        """Calculates a logit boost vector based on the last token emitted."""
        bias = np.zeros(self.tokenizer.vocab_size)
        
        last_word = self.tokenizer.decode([last_token_id]).lower().strip()
        
        if last_word in self.transitions:
            targets = self.transitions[last_word]
            for next_word, boost in targets.items():
                tid = self.tokenizer.word2id.get(next_word, -1)
                if tid != -1 and tid < self.tokenizer.vocab_size:
                    bias[tid] = boost
        
        return bias
