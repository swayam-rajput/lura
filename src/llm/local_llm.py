# src/llm/local_llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from ctransformers import AutoModelForCausalLM
import torch
import threading


class LLM:
    def __init__(self, model_name="microsoft/phi-2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"loading local model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_answer(self, query, retrieved_chunks, max_new_tokens=200):
        context = "\n\n".join(retrieved_chunks)
        print(retrieved_chunks)
        prompt = (
            f"Context:\n{context}\n\nIf the context does not contain the answer, say \"I don't know.\" Do not hallucinate."
            f"Question: {query}\n\n"
            f"Answer concisely and clearly:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return answer
