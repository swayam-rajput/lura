import tiktoken

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    - chunk_size: number of words per chunk
    - overlap: how many words overlap between consecutive chunks
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    num_tokens = len(tokens)

    chunks = []
    start = 0
    
    while start<num_tokens:
        end = min(start + max_tokens, num_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens-overlap
    
    print(f"[Chunks created] {len(chunks)} chunks (~{num_tokens} tokens total).")

    return chunks

if __name__=="__main__":
    words = """Artificial intelligence enables machines to perform tasks that typically require human intelligence. 
    Machine learning is a subset of AI focused on building systems that learn from data. 
    Deep learning, a branch of ML, uses neural networks to model complex patterns.
    """
    chunk_text(words,5,3)