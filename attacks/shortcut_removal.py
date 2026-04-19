from attacks.base import Attack
import random
from utils.text_utils import detokenize, n_modifications, sample_indices, simple_tokenize, word_indices

_SINGLE_CONNECTIVES = {
    "porque",
    "entonces",
    "luego",
}

# Multi-token patterns expressed as lowercase token tuples
_MULTI_CONNECTIVES: list[tuple[str, ...]] = [
    ("por", "eso"),
    ("así", "que"),
    ("asi", "que"),
]

def _find_multi_spans(tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lower = [t.lower() for t in tokens]
    for pat in _MULTI_CONNECTIVES:
        L = len(pat)
        for i in range(0, max(0, len(tokens) - L + 1)):
            if tuple(lower[i : i + L]) == pat:
                spans.append((i, i + L))
    return spans


class ShortcutRemovalAttack(Attack):
    """Remove causal connectives from text, scaled by self.intensity."""

    def _perturb_text(self, text: str) -> str:
        tokens = simple_tokenize(text)
        if not tokens:
            return text

        widx = word_indices(tokens)
        if not widx:
            return text

        spans = _find_multi_spans(tokens)
        single_candidates = [i for i in widx if tokens[i].lower() in _SINGLE_CONNECTIVES]

        units: list[tuple[int, int]] = []
        units.extend(spans)
        units.extend((i, i + 1) for i in single_candidates)

        if not units:
            return text

        n = min(len(units), n_modifications(len(widx), self.intensity))
        chosen_units = sorted(sample_indices(list(range(len(units))), n, rng=random), reverse=True)

        new_tokens = list(tokens)
        for u in chosen_units:
            start, end = units[u]
            for _ in range(end - start):
                if 0 <= start < len(new_tokens):
                    del new_tokens[start]

        return detokenize(new_tokens)
