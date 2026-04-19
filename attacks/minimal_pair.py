import random
from attacks.base import Attack
from utils.text_utils import detokenize, n_modifications, preserve_case, sample_indices, simple_tokenize, word_indices

_NEGATIONS = {
    "no",
    "nunca",
    "jamás",
    "jamas",
    "nadie",
    "ningún",
    "ningun",
    "ninguna",
    "ninguno",
}

_FLIP_MAP = {
    "siempre": "nunca",
    "nunca": "siempre",
}


class MinimalPairAttack(Attack):
    """Create minimal pairs by removing or inverting polarity markers.

    Primary strategy: drop negation tokens (no, nunca, jamás, nadie…) so the
    perturbed sentence has the opposite truth value with minimal surface change.
    Fallback (intensity >= 0.5, no negations found): flip a universal/existential
    quantifier via _FLIP_MAP (siempre ↔ nunca) instead of deleting it.
    """

    def _perturb_text(self, text: str) -> str:
        """Remove n negation tokens (n ∝ intensity); flip a quantifier as fallback."""
        tokens = simple_tokenize(text)
        if not tokens:
            return text

        widx = word_indices(tokens)
        if not widx:
            return text

        candidates = [i for i in widx if tokens[i].lower() in _NEGATIONS]

        new_tokens = list(tokens)
        if not candidates:
            if self.intensity >= 0.5:
                q_candidates = [i for i in widx if tokens[i].lower() in _FLIP_MAP]
                if q_candidates:
                    i = random.choice(q_candidates)
                    new_tokens[i] = preserve_case(new_tokens[i], _FLIP_MAP[new_tokens[i].lower()])
            return detokenize(new_tokens)

        n = min(len(candidates), n_modifications(len(widx), self.intensity))
        chosen = set(sample_indices(candidates, n, rng=random))

        filtered = [t for j, t in enumerate(new_tokens) if j not in chosen]
        return detokenize(filtered)
