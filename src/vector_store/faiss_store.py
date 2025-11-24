import faiss
import numpy as np
import os
import json

INDEX_PATH = "src/faiss/vector_index.faiss"

class FaissStore:
    """
    Local vector database using FAISS + JSON metadata
    """

    def __init__(self, dim: int = 384, index_path: str = INDEX_PATH, model_name:str = "sentence-transformers/all-MiniLM-L12-v2"):
        self.dim = dim
        self.model_name = model_name
        self.index_path = index_path
        self.meta_path = index_path + ".meta"

        self.index = faiss.IndexFlatIP(dim)

        self.documents = []
        self.doc_paths = []
        self.chunk_ids = []

        # Auto-load index + metadata if present
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load_index()

    def add_vectors(self, vectors: np.ndarray, texts: list[str], file_path: str = None, embedder_model:str=None):
        """
        Add vectors + metadata WITHOUT calling load_index() internally.
        """
        if embedder_model and embedder_model != self.model_name:
            raise ValueError(
                f"\n[ERROR] Embedding model mismatch.\n"
                f"Index model: {self.model_name}\n"
                f"New vectors from: {embedder_model}\n"
                f"Run option 3 (Rebuild Index)."
            )
        
        vectors = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"\n[ERROR] Embedding dimension mismatch.\n"
                f"FAISS index dim = {self.dim}, new vectors dim = {vectors.shape[1]}.\n"
                f"This means you changed embedding models.\n"
                f"Run option 3 (Rebuild Index) to fix this safely.\n"
            )
        
        self.index.add(vectors)

        self.documents.extend(texts)
        self.doc_paths.extend([file_path] * len(texts))
        start = len(self.chunk_ids)
        self.chunk_ids.extend(range(start, start + len(texts)))

        print(f"Added {len(vectors)} vectors. Total docs: {len(self.documents)}")


    def save_index(self):
        """
        Save FAISS + JSON metadata properly
        """

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)

        # Build metadata JSON
        metadata = {
            "dim": self.dim,
            "embedding_model": self.model_name,
            "count": len(self.documents),
            "chunks": []
        }

        for i, text in enumerate(self.documents):
            metadata["chunks"].append({
                "id": i,
                "text": text,
                "doc_path": self.doc_paths[i],
                "chunk_id": self.chunk_ids[i]
            })

        # Save JSON
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)


    def load_index(self):
        """
        Safely load FAISS + JSON metadata
        """

        # CASE 1: No FAISS file -> start fresh
        if not os.path.exists(self.index_path):
            print("[INFO] No index found. Creating empty index.")
            self.index = faiss.IndexFlatIP(self.dim)
            self.documents, self.doc_paths, self.chunk_ids = [], [], []
            return

        # CASE 2: FAISS file exists: try loading
        try:
            self.index = faiss.read_index(self.index_path)
        except Exception:
            print("[WARN] FAISS index corrupted. Resetting index.")
            self.index = faiss.IndexFlatIP(self.dim)
            self.documents, self.doc_paths, self.chunk_ids = [], [], []
            return

        if not os.path.exists(self.meta_path):
            print("[WARN] Metadata missing → starting fresh.")
            self.documents, self.doc_paths, self.chunk_ids = [], [], []
            return
        elif os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                self.dim = meta.get("dim", self.dim)
                chunks = meta.get("chunks", [])
                self.model_name = meta.get("embedding_model",self.model_name)
                self.documents = [c["text"] for c in chunks]
                self.doc_paths = [c["doc_path"] for c in chunks]
                self.chunk_ids = [c["chunk_id"] for c in chunks]

                if len(self.documents) != self.index.ntotal:
                    print("[WARN] Metadata count mismatch. Resetting index.")
                    self.index = faiss.IndexFlatIP(self.dim)
                    self.documents, self.doc_paths, self.chunk_ids = [], [], []
            except Exception:
                print("[WARN] Metadata corrupted. Resetting index.")
                self.index = faiss.IndexFlatIP(self.dim)
                self.documents, self.doc_paths, self.chunk_ids = [], [], []

    

    def search_vectors(self, query_vector: np.ndarray, k: int = 3):
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)

        s = scores[0]      # flatten to 1D
        idxs = indices[0]

        if len(s) == 0:
            return []

        top = float(s[0])
        second = float(s[1]) if len(s) > 1 else 0.0
        if top < 0.15:          
            return []
        print('score of result',top)

        if abs(top - second) < 0.005:
            return []
        print('abs')

        results = []
        for idx, score in zip(idxs, s):
            if score < 0.15:
                continue
            if 0 <= idx < len(self.documents):
                results.append({
                    "id": idx,
                    "text": self.documents[idx],
                    "doc_path": self.doc_paths[idx],
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(score)
                })
        print('results >',results)
        return results

    def get_index_path(self):
        return self.index_path
    
    def reset_index(self):
        """Completely clears index + documents."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents = []
        self.doc_paths = []
        self.chunk_ids = []
        faiss.write_index(self.index, self.index_path)
        open(self.meta_path, "w").close()
