from __future__ import annotations
from validation.semantic import check_semantic_similarity
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

MAX_WORKERS = 8  # concurrent Ollama requests


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
        """Return a shallow copy of example with FIELDS_TO_PERTURB rewritten; all other keys unchanged."""
        result = dict(example)
        for field in self.FIELDS_TO_PERTURB:
            if field not in example:
                continue
            
            value = example[field]
            
            if isinstance(value, list):
                new_list = []
                for item in value:
                    perturbed_item = self._perturb_text(item)
                    if check_semantic_similarity(item, perturbed_item):
                        new_list.append(perturbed_item)
                    else:
                        new_list.append(item)
                result[field] = new_list
            else:
                perturbed_val = self._perturb_text(value)
                is_valid = check_semantic_similarity(value, perturbed_val)
                
                print(f"¿Tiene sentido?: {is_valid} | Original: '{value[:30]}...' -> Nuevo: '{perturbed_val[:30]}...'")
                
                if is_valid:
                    result[field] = perturbed_val

        return result

    def apply_dataset(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply the attack to all examples in parallel."""
        results: list[dict[str, Any]] = [{}] * len(dataset)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.apply, ex): i for i, ex in enumerate(dataset)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results
