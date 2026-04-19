from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Attack(ABC):
    """Abstract base class for all lexical robustness attacks."""

    def __init__(self, intensity: float = 0.3):
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")
        self.intensity = intensity

    @abstractmethod
    def apply(self, example: dict[str, Any]) -> dict[str, Any]:
        """Apply the attack to a single dataset example.

        Args:
            example: dict with at least 'text' and 'label' keys.

        Returns:
            New dict with perturbed 'text' and original 'label'.
        """
        ...

    def apply_dataset(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply attack to every example in a dataset."""
        return [self.apply(ex) for ex in dataset]
