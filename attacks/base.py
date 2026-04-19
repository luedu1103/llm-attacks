from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Attack(ABC):
    """Abstract base class for all lexical robustness attacks.

    Subclasses implement `_perturb_text(text) -> str`. The base class
    applies it to every field listed in `FIELDS_TO_PERTURB`, handling
    both plain strings and lists of strings. `answer` and unlisted fields
    are never modified.

    To target different fields, override `FIELDS_TO_PERTURB` in the subclass:
        FIELDS_TO_PERTURB = ["question", "options"]
    """

    FIELDS_TO_PERTURB: list[str] = ["question"]

    def __init__(self, intensity: float = 0.3):
        if not 0.0 <= intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")
        self.intensity = intensity

    @abstractmethod
    def _perturb_text(self, text: str) -> str:
        """Perturb a single string using self.intensity."""
        ...

    def apply(self, example: dict[str, Any]) -> dict[str, Any]:
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
        return [self.apply(ex) for ex in dataset]
