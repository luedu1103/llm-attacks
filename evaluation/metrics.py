from typing import Any


def accuracy(predictions: list[Any], labels: list[Any]) -> float:
    """Fraction of predictions that match ground-truth labels."""
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length")
    if not predictions:
        return 0.0
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def delta_accuracy(
    original_preds: list[Any],
    perturbed_preds: list[Any],
    labels: list[Any],
) -> float:
    """Accuracy drop between original and perturbed predictions.

    Returns original_accuracy - perturbed_accuracy.
    A positive value means the attack degraded performance.
    """
    return accuracy(original_preds, labels) - accuracy(perturbed_preds, labels)


def flip_rate(original_preds: list[Any], perturbed_preds: list[Any]) -> float:
    """Fraction of examples where the model's prediction changed after perturbation."""
    if len(original_preds) != len(perturbed_preds):
        raise ValueError("original_preds and perturbed_preds must have the same length")
    if not original_preds:
        return 0.0
    flips = sum(o != p for o, p in zip(original_preds, perturbed_preds))
    return flips / len(original_preds)
