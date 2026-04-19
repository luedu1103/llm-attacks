from typing import Any
from attacks.base import Attack


class SynonymAttack(Attack):
    """Replace words with synonyms at the given intensity rate.

    Placeholder — returns example unchanged until implemented.
    """

    def apply(self, example: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement synonym substitution using WordNet / LLM
        return dict(example)
