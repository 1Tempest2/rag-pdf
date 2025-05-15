from typing import List
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions

# === Config ===
MAX_TOKENS = 256
MIN_TOKENS = 128
MODEL_NAME = "NYTK/sentence-transformers-experimental-hubert-hungarian"

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

cluster_chunker = ClusterSemanticChunker(
    embedding_function=embedding_function,
    max_chunk_size=MAX_TOKENS,
    min_chunk_size=MIN_TOKENS
)

def chunk_text(text: str) -> List[str]:
    chunks = cluster_chunker.split_text(text)
    return chunks

def llm_tagging(chunk: str) -> str:
    prompt = f"Add 3–5 keywords to this Hungarian paragraph. Return them like [kulcsszó1] [kulcsszó2]:\n\n{chunk}"
    # response = call_your_llm(prompt)
    # return chunk + " " + response
    return chunk  # Placeholder until LLM is integrated
