
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    """
    Handles embedding generation for text using a local model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2"):
        # Load the model locally (no API calls)
        print(f"[EmbeddingModel] Loading model: {model_name}")
        local_path = './models/embeddings/all-MiniLM-L12-v2'
        self.model = SentenceTransformer(local_path,local_files_only=True)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Takes a list of strings and returns a NumPy array of embeddings.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string for vector search.
        """
        return self.embed_texts([query])


