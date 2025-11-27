from encoder.embedder import EmbeddingModel
from storage.faiss_store import FaissStore
import numpy as np


class Retriever:
    _instance = None

    def __init__(self, index_path='src/faiss/vector_index.faiss',top_k:int = 5,chunks=None):
        self.index_path = index_path
        self.top_k = top_k
        
        self.embedder = EmbeddingModel()
        try:
            self.dim = self.embedder.dim
        except:
            self.dim = 384
        
        self.store = FaissStore(self.dim,index_path=index_path)
        try:
            self.store.load_index()
        except:
            print('[Retriever] Warning: Index failed to load')
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.store = FaissStore()
        return cls._instance
    

    def _embed(self, text:str)-> np.ndarray:
        """Embed a query string"""
        if hasattr(self.embedder,"embed_query"):
            vec = self.embedder.embed_query(text)
        else:
            vec = self.embedder.embed_texts([text])
        
        vec = np.asarray(vec,dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1,-1)
        return vec
    

    def search(self, query:str, k:int=None):
        """Embed query, search FAISS, returns top-k text chunks"""
        k = k or self.top_k

        if not self.store.index or self.store.index.ntotal == 0:
            print('[Retriever Error]: No vectors inside FAISS index.')
            return []
        
        query_vector = self._embed(query)
        results = self.store.search_vectors(query_vector,k=k)
        
        # results = [(self.chunks[i],float(scores[0][j])) for j, i in enumerate(ids[0])]
        return results
        