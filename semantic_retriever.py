from typing import List
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Custom Retriever class
class SemanticRetriever:
    def __init__(self, model_name: str = "NYTK/sentence-transformers-experimental-hubert-hungarian"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_db = None

    def index_documents(self, documents: List[Document]):
        """Index pre-chunked documents."""
        self.vector_db = FAISS.from_documents(documents, self.embeddings)

    def retrieve(self, query: str, k: int = 10):
        """Retrieve top-k relevant chunks using keyword-enhanced semantic search."""
        return self.vector_db.similarity_search(query, k=k)

