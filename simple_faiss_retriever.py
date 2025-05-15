from typing import List, Tuple
import faiss
from sentence_transformers import SentenceTransformer

class FaissRetriever:
    def __init__(self, dimension: int, model_name: str = "NYTK/sentence-transformers-experimental-hubert-hungarian"):
        self.dimension = dimension
        self.embedding_model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatIP(dimension)
        self.text_chunks = []

    def embed_documents(self, texts):
        return self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def embed_query(self, text):
        return self.embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)

    def add_texts(self, texts: List[str]):
        embeddings = self.embed_documents(texts)
        self.index.add(embeddings)
        self.text_chunks.extend(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_embedding = self.embed_query(query)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.text_chunks[idx], dist))
        return results
