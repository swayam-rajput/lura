from llama_cpp import Llama

class LLM:
    def __init__(self, model_path="models/model.gguf"):
        print("Loading GGUF model...")
        self.llm = Llama(
            model_path=model_path,
            n_threads=8,
            n_ctx=4096,
            chat_format='qwen'
        )
        print("Model loaded.")
    
    def _build_prompt(self,question, chunks):
        context = ''
        for i, c in enumerate(chunks):
            clean = c['text'].replace('\n',' ')
            context += f'[Chunk {i}] {clean}\n\n'
        
        system_prompt = (
            "You are a retrieval based assistant."
            "Use ONLY the provided context to answer"
            "If the answer is not in the context, reply exactly: 'I don't know, its not in the context index, provide me with appropriate context'"
        )

        user_prompt = (
            f"### Context ###\n{context}\n"
            f"### Question ###\n{question}\n"
            "### Answer ###\n"
        )

        return [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ]

    def generate(self, question, chunks, max_new_tokens=200):
        messages = self._build_prompt(question,chunks)
        result = self.llm.create_chat_completion(messages, max_tokens=max_new_tokens, temperature=0.0)
        return result["choices"][0]["message"]["content"].strip()
