from typing import Any


def check_semantic_similarity(
    original: str,
    perturbed: str,
    threshold: float = 0.85,
) -> bool:
    """Return True if original and perturbed texts are semantically similar enough.

    Placeholder — always returns True until a real similarity model is wired in.

    Args:
        original: Source text before perturbation.
        perturbed: Text after perturbation.
        threshold: Minimum cosine similarity to accept (0–1).
    """
    # TODO: implement using sentence-transformers or similar
    return True
