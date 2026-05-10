from typing import Any
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def check_semantic_similarity(
    original: str,
    perturbed: str,
    threshold: float = 0.85,
) -> bool:
    """Return True if original and perturbed texts are semantically similar enough."""
    
    if not original or not perturbed:
        return False

    embeddings1 = model.encode(original, convert_to_tensor=True)
    embeddings2 = model.encode(perturbed, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    return cosine_scores.item() >= threshold
