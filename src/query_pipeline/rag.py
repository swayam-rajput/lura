from .retrieve import Retriever
from llm.local_llm import LLM

def run_rag(question:str):
    retriever = Retriever()

    chunks = retriever.search(question,k=5)

    context = ''
    for i,c in enumerate(chunks,start=1):
        clean = c['text'].replace('\n',' ')
        context += f"[{i}] {clean}\n\n"
    
    
    prompt = (
        "Use ONLY the context below to answer the question.\n"
        "If answer is not in context, say 'I don't know.'\n\n"
        f"### CONTEXT ###\n{context}\n"
        f"### QUESTION ###\n{question}\n"
        "### ANSWER ###\n"
    )

    llm = LLM()
    response = llm.generate(question,chunks)
    
    
    return response,chunks
    
