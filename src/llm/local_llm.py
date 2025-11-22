# src/llm/local_llm.py
from llama_cpp import Llama

class LLM:
    def __init__(self, model_path="models/model.gguf"):
        print("Loading GGUF model...")
        self.llm = Llama(
            model_path=model_path,
            n_threads=8,
            n_ctx=4096
        )
        print("Model loaded.")

    def generate(self, query, retrieved_chunks, max_new_tokens=200):
        context = "\n\n".join(ch["text"] for ch in retrieved_chunks)
        prompt = (
            f"Use the context to answer. If answer isn't in context, say 'I don't know.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        result = self.llm(prompt, max_tokens=max_new_tokens)
        return result["choices"][0]["text"].strip()
