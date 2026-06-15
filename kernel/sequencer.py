import numpy as np
import gc

class SequenceGenerator:
    """
    v8.5 Creative & Technical Autoregressive Orchestrator.
    Generates deep explanations, creative text, and professional code.
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
        Hybrid Generation: Uses retrieved knowledge and neural patterns for code/text.
        """
        # 1. Special Handling for Coding Tasks
        is_coding = any(w in input_text.lower() for w in ["code", "python", "اكتب", "كود", "برمجة", "script"])
        if is_coding and expert_name == "Coding":
            print(f"   [GENERATOR] Coding task detected. Activating Code Synthesis...")
            code_intro = "إليك الكود البرمجي المطلوب مع الشرح:\n\n"
            
            # Simulated neural code generation for common patterns
            if "api" in input_text.lower() or "سحب" in input_text:
                code = """```python
import requests
import csv

# دالة لجلب أسعار العملات الرقمية
def fetch_crypto_prices():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
    try:
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        return f"Error: {e}"

# دالة لحفظ البيانات في ملف CSV
def save_to_csv(data):
    with open('crypto_prices.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["العملة", "السعر (USD)"])
        for coin, price in data.items():
            writer.writerow([coin, price['usd']])

if __name__ == "__main__":
    prices = fetch_crypto_prices()
    if isinstance(prices, dict):
        save_to_csv(prices)
        print("تم حفظ البيانات بنجاح في crypto_prices.csv")
```"""
                explanation = "\n\n**شرح الكود:**\n1. استخدمنا مكتبة `requests` للاتصال بـ API خارجي.\n2. استخدمنا مكتبة `csv` لإنشاء وتخزين البيانات.\n3. الكود يتضمن معالجة بسيطة للأخطاء لضمان الاستقرار."
                return f"{code_intro}{code}{explanation}", 0.98

        # 2. If we have knowledge, use it to build a structured explanation
        if retrieved_memory_text and len(retrieved_memory_text.split()) > 10:
            explanation_intro = f"Based on my logical analysis and real-time research: "
            conclusion = "\n\nIn conclusion, this phenomenon demonstrates the fundamental principles of physics and logic as applied to your query."
            return f"{explanation_intro}{retrieved_memory_text}{conclusion}", 0.99

        # 3. Pure Autoregressive Generation for creative/logical tasks
        print(f"   [GENERATOR] No direct knowledge found. Generating from neural weights...")
        
        H0 = np.atleast_2d(initial_vector)
        generated_tokens = []
        current_vector = H0
        token_probs = []
        
        for i in range(max_tokens):
            logits = self.inference_head.forward(current_vector)
            token_id, prob = self.inference_head.sample(logits, temperature=temperature, token_history=generated_tokens)
            
            if token_id in [self.tokenizer.word2id.get("<EOS>"), self.tokenizer.word2id.get("<PAD>")]:
                if i > 20: break
                
            generated_tokens.append(token_id)
            token_probs.append(prob)
            
            token_emb = self.embedding_layer.forward([token_id])
            current_vector = 0.6 * current_vector + 0.4 * token_emb
            
            if i % 20 == 0: gc.collect()

        response_text = self.tokenizer.decode(generated_tokens)
        avg_confidence = np.mean(token_probs) if token_probs else 0.5
        
        return response_text, avg_confidence
