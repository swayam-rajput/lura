from embeddings.embedder import EmbeddingModel
from vector_store.faiss_store import FaissStore

class Retriever:
    def __init__(self, store_path='src/vector_index.faiss',chunks=None):
        self.store = FaissStore(dim=384,index_path=store_path)
        self.chunks = chunks
    
    def search(self,query, k=3):
        """Embed query, search FAISS, returns top-k text chunks"""
        if not self.store:
            print('Error: Index not loaded.')
            return None
        query_vector = EmbeddingModel().embed_query(query)
        results = self.store.search_vectors(query_vector,k=k)
        
        # results = [(self.chunks[i],float(scores[0][j])) for j, i in enumerate(ids[0])]
        return results
        