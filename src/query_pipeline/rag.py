from .retrieve import Retriever
from llm.local_llm import LLM

def run_rag(question:str):
    retriever = Retriever()

    chunks = retriever.search(question,k=5)

    llm = LLM()
    response = llm.generate(question,chunks)
    
    
    return response,chunks
    
