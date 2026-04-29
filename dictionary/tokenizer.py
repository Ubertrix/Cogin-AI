import os
import re
import json
import numpy as np

class Tokenizer:
    """Advanced Linguistic Analyzer - DIRECT LOADING from Dictionary"""
    def __init__(self, vocab_path=None):
        if vocab_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.vocab_path = os.path.join(base_dir, "dictionary", "vocab.json")
        else:
            self.vocab_path = vocab_path
            
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        
        # Load DIRECTLY from vocab.json (26,401 tokens)
        self.load_vocab()
        
        self.symbol_pattern = r'([+\-*/=<>!&|(){}\[\];.,])'
        
        # Initialize embedding matrix directly from vocab size
        self._init_embedding_matrix()

    def _init_embedding_matrix(self):
        """Initialize embedding matrix (d_model x vocab_size)"""
        self.d_model = 1024  # Standard dimension
        # Random init - will be replaced by brain_weights
        self.embeddings = np.random.randn(self.vocab_size, self.d_model) * 0.01

    def save_vocab(self):
        """Save word map to JSON file to ensure persistence after restart."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
            data = {"word2id": self.word2id, "vocab_size": self.vocab_size}
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"   [Error] Could not save vocab: {e}")

    def load_vocab(self):
        """Load the saved dictionary."""
        if os.path.exists(self.vocab_path):
            try:
                import json
                with open(self.vocab_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.word2id = data.get("word2id", self.word2id)
                self.vocab_size = data.get("vocab_size", len(self.word2id))
                self.id2word = {int(v): k for k, v in self.word2id.items()}
                print(f"Dictionary Loaded: {self.vocab_size} tokens known.")
            except Exception as e:
                print(f"   [Error] Could not load vocab: {e}")

    def _tokenize(self, text):
        """Convert text to word list with symbol separation"""
        # Separate mathematical and programming symbols from words
        text = re.sub(self.symbol_pattern, r' \1 ', text)
        # Split text based on spaces and convert to lowercase
        return text.lower().split()

    def _infer_subword(self, word):
        """Inference-Based Fallback logic: break an unknown word into character-level tokens (Sub-word inference)"""
        sub_tokens = []
        for char in word:
            # prefix with '##' to denote it's a subword/character piece, if we wanted to
            # for simplicity, we just use the characters themselves
            # if the char is not in vocab, we really have to use <UNK>
            sub_tokens.append(char)
        return sub_tokens

    def fit(self, sentences):
        """Build dictionary from a large dataset"""
        for sentence in sentences:
            if isinstance(sentence, str):
                words = self._tokenize(sentence)
                for word in words:
                    if word not in self.word2id:
                        self.word2id[word] = self.vocab_size
                        self.id2word[self.vocab_size] = word
                        self.vocab_size += 1
                        
                    # Only learn characters if they are Arabic or alphanumeric to avoid noise
                    for char in word:
                        if char not in self.word2id:
                            if re.match(r'^[\u0600-\u06FF]$', char) or char.isalnum():
                                self.word2id[char] = self.vocab_size
                                self.id2word[self.vocab_size] = char
                                self.vocab_size += 1
        print(f"Multi-Language Dictionary Updated: {self.vocab_size} tokens known.")
        self.save_vocab() # Persistence Trigger

    def encode(self, text):
        """Convert text to numbers via Sub-word Inference logic"""
        words = self._tokenize(text)
        token_ids = []
        for word in words:
            if word in self.word2id:
                token_ids.append(self.word2id[word])
            else:
                # Infinite Dictionary Smoothing: Character-level inference
                # We break the word down and ensure every char is mapped to a token
                sub_words = self._infer_subword(word)
                for sub in sub_words:
                    if sub in self.word2id:
                         token_ids.append(self.word2id[sub])
                    else:
                         token_ids.append(self.word2id.get("<UNK>", 1))
        return token_ids

    def decode(self, token_ids):
        """Convert numbers back to text"""
        return " ".join([self.id2word.get(tid, "<UNK>") for tid in token_ids])

    def load_embeddings_from_brain(self, brain_weights_path=None):
        """DIRECT LOADING: Initialize embeddings from brain_weights.npy"""
        if brain_weights_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            brain_weights_path = os.path.join(base_dir, "brain_weights.npy")
        
        if not os.path.exists(brain_weights_path):
            print("   [Tokenizer] brain_weights.npy not found, using random init")
            return
            
        try:
            data = np.load(brain_weights_path, allow_pickle=True).item()
            
            # Build embedding matrix from brain weights
            concepts = list(data.keys())
            print(f"   [Tokenizer] Loading {len(concepts)} anchors from brain_weights...")
            
            # Initialize with proper dimension
            self.embeddings = np.random.randn(self.vocab_size, self.d_model) * 0.01
            
            for i, concept in enumerate(concepts[:self.vocab_size]):
                entry = data[concept]
                if "essence" in entry:
                    vec = entry["essence"]
                    if hasattr(vec, 'shape') and len(vec.shape) > 0:
                        # Map to embedding
                        vec_flat = vec.flatten()[:self.d_model]
                        if i < self.vocab_size:
                            self.embeddings[i] = vec_flat
            
            print(f"   [Tokenizer] Embeddings initialized: {self.embeddings.shape}")
        except Exception as e:
            print(f"   [Tokenizer Error] {e}")
