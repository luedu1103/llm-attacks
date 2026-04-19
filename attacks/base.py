from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Attack(ABC):
    """Abstract base class for all lexical robustness attacks.

    Each concrete attack only needs to implement _perturb_text(text) -> str.
    The base class applies it to the relevant fields: 'question' and each
    option in 'options'. The 'answer' index is never modified.
    """

    # Subclasses can override to restrict which fields are perturbed
    FIELDS_TO_PERTURB: list[str] = ["question", "options"]

    def __init__(self, intensity: float = 0.3):
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")
        self.intensity = intensity

    @abstractmethod
    def _perturb_text(self, text: str) -> str:
        """Perturb a single string. Implemented by each concrete attack."""
        ...

    def apply(self, example: dict[str, Any]) -> dict[str, Any]:
        """Apply the attack to a single dataset example.

        Perturbs 'question' (str) and each item in 'options' (list[str]).
        'answer' and all other fields are copied unchanged.
        """
        result = dict(example)
        for field in self.FIELDS_TO_PERTURB:
            if field not in example:
                continue
            value = example[field]
            if isinstance(value, list):
                result[field] = [self._perturb_text(item) for item in value]
            else:
                result[field] = self._perturb_text(value)
        return result

    def apply_dataset(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply attack to every example in a dataset."""
        return [self.apply(ex) for ex in dataset]
