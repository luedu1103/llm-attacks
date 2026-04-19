from attacks.base import Attack
import random
import re
from utils.text_utils import normalize_whitespace, n_modifications


_SUBS: list[tuple[str, str]] = [
    (r"\bdecidió\b", "optó"),
    (r"\bdecidio\b", "opto"),
    (r"\ben silencio\b", "calladamente"),
    (r"\bantes de\b", "previo a"),
    (r"\bpor eso\b", "por esa razón"),
    (r"\basí que\b", "de modo que"),
    (r"\basi que\b", "de modo que"),
    (r"\btal vez\b", "quizá"),
    (r"\bquizá\b", "tal vez"),
    (r"\bpuede que\b", "es posible que"),
    (r"\bporque\b", "ya que"),
    (r"\bluego\b", "después"),
    (r"\bentonces\b", "en consecuencia"),
]


def _apply_some_subs(text: str, k: int) -> str:
    """Apply at most k regex substitutions from _SUBS in random order."""
    if not text or k <= 0:
        return text
    subs = list(_SUBS)
    random.shuffle(subs)
    changed = text
    applied = 0
    for pat, rep in subs:
        if applied >= k:
            break
        if re.search(pat, changed, flags=re.IGNORECASE):
            changed = re.sub(pat, rep, changed, flags=re.IGNORECASE)
            applied += 1
    return changed


class ParaphraseAttack(Attack):
    """Surface-level paraphrase via lexical and connective substitution.

    At low-to-medium intensity: replaces connectives and common phrases with
    functionally equivalent alternatives (e.g. 'porque' → 'ya que').
    At high intensity (>= 0.5): additionally swaps the two clauses around the
    first semicolon if one is present, altering surface order without changing
    the propositional content.
    """

    def _perturb_text(self, text: str) -> str:
        """Apply k substitutions (k ∝ intensity) then optionally reorder clauses."""
        if not text.strip():
            return text

        n_ops = n_modifications(num_words=max(4, len(text.split())), intensity=self.intensity, min_n=1)
        n_ops = min(6, max(1, n_ops))

        rewritten = _apply_some_subs(text, k=n_ops)

        if self.intensity >= 0.5 and ";" in rewritten:
            parts = [p.strip() for p in rewritten.split(";", maxsplit=1)]
            if len(parts) == 2 and all(parts):
                rewritten = parts[1].rstrip(".") + ". " + parts[0].rstrip(".") + "."

        return normalize_whitespace(rewritten)
