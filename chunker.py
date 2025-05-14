from typing import List
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions

# === Config ===
MAX_TOKENS = 256
MODEL_NAME = "NYTK/sentence-transformers-experimental-hubert-hungarian"

# You can replace this with your preferred embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

# Initialize the semantic cluster chunker
cluster_chunker = ClusterSemanticChunker(
    embedding_function=embedding_function,
    max_chunk_size=MAX_TOKENS  # ClusterSemanticChunker uses token length estimates
)

def chunk_text(text: str) -> List[str]:
    """
    Chunks the input text using ClusterSemanticChunker from chunking_evaluation.
    """
    chunks = cluster_chunker.split_text(text)
    return chunks

# Optional: add post-processing or keyword tagging if needed
def llm_tagging(chunk: str) -> str:
    prompt = f"Add 3–5 keywords to this Hungarian paragraph. Return them like [kulcsszó1] [kulcsszó2]:\n\n{chunk}"
    # response = call_your_llm(prompt)
    # return chunk + " " + response
    return chunk  # Placeholder until LLM is integrated
