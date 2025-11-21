import faiss
import numpy as np
import os
import json

INDEX_PATH = "src/vector_index.faiss"

class FaissStore:
    """
    Local vector database using FAISS + JSON metadata
    """

    def __init__(self, dim: int = 384, index_path: str = INDEX_PATH):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = index_path + ".meta"

        self.index = faiss.IndexFlatIP(dim)

        self.documents = []
        self.doc_paths = []
        self.chunk_ids = []

        # Auto-load index + metadata if present
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load_index()

    def add_vectors(self, vectors: np.ndarray, texts: list[str], file_path: str = None):
        """
        Add vectors + metadata WITHOUT calling load_index() internally.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        self.documents.extend(texts)
        self.doc_paths.extend([file_path] * len(texts))
        self.chunk_ids.extend(list(range(len(texts))))

        print(f"Added {len(vectors)} vectors. Total docs: {len(self.documents)}")


    def save_index(self):
        """
        Save FAISS + JSON metadata properly
        """

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Build metadata JSON
        metadata = {
            "dim": self.dim,
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


    def load_index(self):
        """
        Load FAISS + JSON metadata
        """

        if not os.path.exists(self.index_path):
            print("Index file not found. Creating a new empty index.")
            self.index = faiss.IndexFlatIP(self.dim)
            self.documents = []
            self.doc_paths = []
            self.chunk_ids = []
            return

        # Load FAISS
        self.index = faiss.read_index(self.index_path)

        # Load metadata JSON
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.dim = meta["dim"]
            self.documents = [c["text"] for c in meta["chunks"]]
            self.doc_paths = [c["doc_path"] for c in meta["chunks"]]
            self.chunk_ids = [c["chunk_id"] for c in meta["chunks"]]
        else:
            print("Metadata missing — starting fresh.")
            self.documents = []
            self.doc_paths = []
            self.chunk_ids = []


    def search_vectors(self, query_vector: np.ndarray, k: int = 3):
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)

        s = scores[0]      # flatten to 1D
        idxs = indices[0]

        if len(s) == 0:
            return []

        top = float(s[0])
        second = float(s[1]) if len(s) > 1 else 0.0

        if top < 0.22:          
            return []

        if abs(top - second) < 0.03:
            return []

        results = []
        for idx, score in zip(idxs, s):
            if score < 0.22:
                continue
            if 0 <= idx < len(self.documents):
                results.append({
                    "id": idx,
                    "text": self.documents[idx],
                    "doc_path": self.doc_paths[idx],
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(score)
                })

        return results


    def get_index_path(self):
        return self.index_path
