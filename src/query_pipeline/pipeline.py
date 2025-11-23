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
