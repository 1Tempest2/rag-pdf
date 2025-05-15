import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import os
from pathlib import Path

class HungarianRetriever:
    def __init__(self, collection_name: str = "hungarian_documents"):
        """Initialize the retriever with a ChromaDB collection"""
        # Use same embedding model as in chunker.py
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="NYTK/sentence-transformers-experimental-hubert-hungarian"
        )
        
        # Initialize ChromaDB client
        persist_directory = Path("chroma_db")
        persist_directory.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_functionK
            )
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
    
    def add_chunks(self, chunks: List[str], document_id: str) -> None:
        """Add chunks to the vector database"""
        # Create chunk IDs
        chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        # Create metadata for each chunk
        metadata_list = [{"source": document_id, "chunk_index": i} for i in range(len(chunks))]
        
        # Add chunks to collection
        self.collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=metadata_list
        )
        
        print(f"Added {len(chunks)} chunks from document {document_id} to vector database")
    
    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on user query"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "chunk": doc,
                    "metadata": metadata,
                    "relevance_score": 1 - distance,  # Convert distance to similarity score
                    "rank": i+1
                })
                
            return formatted_results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def delete_collection(self) -> None:
        """Delete the collection - useful for testing"""
        self.client.delete_collection(self.collection.name)
        print(f"Deleted collection {self.collection.name}")
