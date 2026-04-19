from typing import Any
from attacks.base import Attack


class SynonymAttack(Attack):
    """Replace words with synonyms at the given intensity rate.

    Placeholder — returns example unchanged until implemented.
    """

    def _perturb_text(self, text: str) -> str:
        # TODO: implement synonym substitution using WordNet / LLM
        return text
