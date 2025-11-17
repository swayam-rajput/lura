import faiss
import numpy as np 
import os

class FaissStore:
    """
    Local vector database using FAISS
    """
    def __init__(self,dim:int=384,index_path:str='vector_index.faiss'):
        self.dim = dim
        self.index_path = index_path

        self.index = faiss.IndexFlatIP(dim)

        self.documents = []
        
    def add_vectors(self, vectors:np.ndarray,texts:list[str]):
        """Adds vectors and original texts to the index"""
        faiss.normalize_L2(vectors)
        self.load_index()
        self.index.add(vectors)
        self.documents.extend(texts)
        print(f"Added {len(vectors)} vectors. Total docs: {len(self.documents)}")
    

    def search_vectors(self, query_vector:np.ndarray, k:int = 3):
        """Search for top-k most similar vectors and return their texts and scores"""
        faiss.normalize_L2(query_vector)
        self.load_index()
        print('self.index: ',self.index)
        scores, indices = self.index.search(query_vector,k)
        results = []
        print(scores)
        try:
            for i, score in zip(indices[0], scores[0]):
                if i < len(self.documents):
                    results.append({"text": self.documents[i], "score":float(score)})
        except IndexError as ie:
            print(f'IndexError: FAISS returned {len(scores)} results.\n')
        return results

    def save_index(self):
        """Save the FAISS index and corresponding texts"""
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".meta", "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(doc.replace("\n", " ") + "\n")

    def load_index(self):
        """
        Load the FAISS index and metadata.
        """
        print(self.get_index_path())
        print('path: ',self.index_path)
        if not os.path.exists(self.index_path):
            print("Index file not found. Creating new empty index...")
            self.index = faiss.IndexFlatIP(self.dim)
            self.documents = []
            # create empty metadata file so future loads don't fail
            with open(self.index_path + ".meta", "w", encoding="utf-8") as f:
                pass
            faiss.write_index(self.index, self.index_path)
            return

        self.index = faiss.read_index(self.index_path)
        metapath = self.index_path + ".meta"
        if os.path.exists(metapath):
            with open(self.index_path + ".meta", "r", encoding="utf-8") as f:
                self.documents = [line.strip() for line in f.readlines()]
        else:
            self.documents = []




    def get_index_path(self):
        return self.index_path
      
    