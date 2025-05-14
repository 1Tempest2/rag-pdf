from typing import List
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# === Config ===
MAX_TOKENS = 128
OVERLAP_RATIO = 0.3
MODEL_NAME = "NYTK/sentence-transformers-experimental-hubert-hungarian"

model = SentenceTransformer(MODEL_NAME)


def count_tokens(text: str) -> int:
    return len(model.tokenizer.encode(text, add_special_tokens=False))


def split_into_sentences(text: str) -> List[str]:
    return sent_tokenize(text)


def create_semantic_clusters(sentences: List[str]) -> List[List[str]]:
    embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True, batch_size=32)
    clusters, current = [], [sentences[0]]

    sims = [
        util.pytorch_cos_sim(embeddings[i], embeddings[i - 1]).item()
        for i in range(1, len(embeddings))
    ]
    mean_sim = sum(sims) / len(sims)
    std_sim = (sum((x - mean_sim) ** 2 for x in sims) / len(sims)) ** 0.5

    dynamic_threshold = mean_sim - 0.5 * std_sim

    for i in range(1, len(sentences)):
        sim = util.pytorch_cos_sim(embeddings[i], embeddings[i - 1]).item()
        if sim >= dynamic_threshold:
            current.append(sentences[i])
        else:
            clusters.append(current)
            current = [sentences[i]]
    if current:
        clusters.append(current)
    return clusters


def sliding_token_chunks(clusters: List[List[str]], max_tokens: int, overlap: float) -> List[str]:
    final_chunks = []
    for cluster in clusters:
        i = 0
        while i < len(cluster):
            window, tokens, j = [], 0, i
            while j < len(cluster) and tokens < max_tokens:
                sent = cluster[j]
                tokens += count_tokens(sent)
                if tokens <= max_tokens:
                    window.append(sent)
                    j += 1
                else:
                    break
            final_chunks.append(" ".join(window))
            i += max(1, int(len(window) * (1 - overlap)))
    return final_chunks


def llm_tagging(chunk: str) -> str:
    prompt = f"Add 3–5 keywords to this Hungarian paragraph. Return them like [kulcsszó1] [kulcsszó2]:\n\n{chunk}"
    #response = call_your_llm(prompt)  # Your LLM call
    #return chunk + " " + response


def chunk_text(text: str) -> List[str]:
    sentences = split_into_sentences(text)
    clusters = create_semantic_clusters(sentences)
    chunks = sliding_token_chunks(clusters, MAX_TOKENS, OVERLAP_RATIO)
    #return [llm_tagging(chunk) for chunk in chunks]
    return chunks
