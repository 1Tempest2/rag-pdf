import numpy as np
import faiss
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class FAISSHungarianRetriever:
    def __init__(self, 
                 index_directory: str = "faiss_index", 
                 model_name: str = "NYTK/sentence-transformers-experimental-hubert-hungarian"):
        """Initialize the FAISS retriever with a sentence transformer model"""
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Directory for storing FAISS index and metadata
        self.index_directory = Path(index_directory)
        self.index_directory.mkdir(exist_ok=True, parents=True)
        
        self.index_path = self.index_directory / "index.faiss"
        self.meta_path = self.index_directory / "metadata.pkl"
        
        # Initialize or load index and metadata
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, 'rb') as f:
                self.document_chunks = pickle.load(f)
            print(f"Loaded existing index with {self.index.ntotal} vectors")
        else:
            # Initialize a new index
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            self.document_chunks = []
            print(f"Created new FAISS index with dimension {self.embedding_dimension}")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        return self.embedding_model.encode(text)
    
    def add_chunks(self, chunks: List[str], document_id: str) -> None:
        """Add document chunks to the FAISS index"""
        # First store the document metadata with the original text
        current_index = len(self.document_chunks)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "source": document_id,
                "chunk_index": i,
                "text": chunk
            }
            self.document_chunks.append(chunk_metadata)
        
        # Then encode all chunks into embeddings
        embeddings = np.array([self._encode_text(chunk) for chunk in chunks]).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Save the updated index and metadata
        self._save_index()
        
        print(f"Added {len(chunks)} chunks from document {document_id} to FAISS index")
    
    def search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on user query"""
        try:
            # Encode the query
            query_vector = self._encode_text(query).reshape(1, -1).astype('float32')
            
            # Search the index
            distances, indices = self.index.search(query_vector, k=n_results)
            
            # Format results
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx != -1 and idx < len(self.document_chunks):  # Ensure index is valid
                    chunk_data = self.document_chunks[idx]
                    
                    results.append({
                        "chunk": chunk_data["text"],
                        "metadata": {
                            "source": chunk_data["source"],
                            "chunk_index": chunk_data["chunk_index"]
                        },
                        "relevance_score": 1.0 / (1.0 + distance),  # Convert distance to similarity score
                        "rank": i+1
                    })
            
            return results
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return []
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.document_chunks, f)
    
    def clear_index(self) -> None:
        """Clear the index and start fresh"""
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.document_chunks = []
        self._save_index()
        print("FAISS index cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            "vector_dimension": self.embedding_dimension,
            "total_vectors": self.index.ntotal,
            "documents": len(set(chunk["source"] for chunk in self.document_chunks)),
            "chunks": len(self.document_chunks),
            "model": self.model_name
        }

# For better performance with larger datasets, consider using IndexIVFFlat
class FAISSOptimizedRetriever(FAISSHungarianRetriever):
    def __init__(self, 
                 index_directory: str = "faiss_index", 
                 model_name: str = "NYTK/sentence-transformers-experimental-hubert-hungarian",
                 nlist: int = 100):  # Number of Voronoi cells
        """Initialize the optimized FAISS retriever with IVF indexing"""
        super().__init__(index_directory, model_name)
        
        # Override index initialization if it doesn't exist already
        if not self.index_path.exists():
            # For small datasets, use IndexFlatL2
            if nlist < 20:  # Too few clusters can degrade performance
                self.index = faiss.IndexFlatL2(self.embedding_dimension)
                print(f"Created FlatL2 index (dataset too small for IVF)")
            else:
                # Create quantizer
                quantizer = faiss.IndexFlatL2(self.embedding_dimension)
                # Create IVF index - nlist is the number of centroids
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
                self.index.nprobe = min(20, nlist // 5)  # Number of cells to probe at search time
                print(f"Created optimized IVFFlat index with {nlist} clusters")
                
                # Index needs training before adding vectors
                self.needs_training = True
        else:
            self.needs_training = False
    
    def add_chunks(self, chunks: List[str], document_id: str) -> None:
        """Add document chunks with training if needed"""
        # First encode all chunks into embeddings
        embeddings = np.array([self._encode_text(chunk) for chunk in chunks]).astype('float32')
        
        # Train the index if needed and this is the first insertion
        if hasattr(self, 'needs_training') and self.needs_training and len(chunks) > 0:
            if not self.index.is_trained:
                print(f"Training the IVF index with {len(embeddings)} vectors...")
                self.index.train(embeddings)
                self.needs_training = False
        
        # Store metadata
        current_index = len(self.document_chunks)
        for i, chunk in enumerate(chunks):
            self.document_chunks.append({
                "source": document_id,
                "chunk_index": i,
                "text": chunk
            })
        
        # Add to index
        self.index.add(embeddings)
        
        # Save
        self._save_index()
        print(f"Added {len(chunks)} chunks from document {document_id} to FAISS index")
