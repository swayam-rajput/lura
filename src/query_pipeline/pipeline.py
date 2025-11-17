# This file is optional 


# The **`query_pipeline`** folder controls how a user query travels through your system — embedding, retrieval, and response generation. It’s the brain that connects embeddings, FAISS search, and (optionally) your local LLM.

# Here’s what each file does and how to build it:

# ---

# ### **1. `__init__.py`**

# Purpose: marks the folder as a Python package.
# Keep it empty or just include:

# ```py
# # src/query_pipeline/__init__.py
# ```

# ---

# ### **2. `retriever.py`**

# Purpose: handle the full **retrieval flow** — from query text → vector → top results.
# This file depends on two modules:

# * `embeddings/embedder.py` (for creating the query vector)
# * `vectorstore/faiss_store.py` (for searching stored vectors)

# **Example implementation:**

# ```py
# # src/query_pipeline/retriever.py
# from src.embeddings.embedder import embed_texts
# from src.vectorstore.faiss_store import FaissStore

# class Retriever:
#     def __init__(self, store_path="vector_index.faiss", chunks=None):
#         self.store = FaissStore(dim=384, path=store_path)  # 384 = MiniLM dimension
#         self.chunks = chunks  # list of text chunks corresponding to embeddings

#     def search(self, query, k=3):
#         """Embed query, search FAISS, return top-k text chunks."""
#         query_vec = embed_texts([query])
#         ids, scores = self.store.search(query_vec, k=k)
#         results = [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(ids[0])]
#         return results
# ```

# **How it works:**

# 1. `embed_texts()` turns the query into a vector.
# 2. `FaissStore.search()` finds top-k nearest vectors.
# 3. Returns the corresponding text chunks with similarity scores.

# ---

# ### Optional extension: `pipeline.py`

# Purpose: tie retrieval + LLM together once you add your **local LLM**.
# Example:

# ```py
# # src/query_pipeline/pipeline.py
# from src.query_pipeline.retriever import Retriever
# from src.llm.local_llm import LocalLLM

# class QueryPipeline:
#     def __init__(self, retriever: Retriever, llm: LocalLLM):
#         self.retriever = retriever
#         self.llm = llm

#     def answer(self, query):
#         """Retrieve top chunks and generate final answer."""
#         results = self.retriever.search(query, k=3)
#         context = "\n\n".join([r[0] for r in results])
#         return self.llm.generate_answer(query, context)
# ```

# This lets you later call:

# ```py
# pipeline = QueryPipeline(retriever, llm)
# response = pipeline.answer("What is anomaly detection?")
# print(response)
# ```

# ---

# ### Summary of flow

# ```
# User query
#    ↓
# embed_texts() → vector
#    ↓
# FaissStore.search() → top chunks
#    ↓
# LLM (optional) → contextual answer
#    ↓
# output
# ```

# ---

# Do you want me to now walk you through **the FAISS store files next** (so the retriever has a backend to search)?


# src/query_pipeline/pipeline.py
from query_pipeline.retrieve import Retriever
from src.llm.local_llm import LLM

class QueryPipeline:
    def __init__(self, retriever: Retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query):
        """Retrieve top chunks and generate final answer."""
        results = self.retriever.search(query, k=3)
        context = "\n\n".join([r[0] for r in results])
        return self.llm.generate_answer(query, context)


# pipeline = QueryPipeline(retriever, llm)
# response = pipeline.answer("What is anomaly detection?")
# print(response)
