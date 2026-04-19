from typing import Any
from attacks.base import Attack


class ParaphraseAttack(Attack):
    """Paraphrase sentences while preserving meaning.

    Placeholder — returns example unchanged until implemented.
    """

    def _perturb_text(self, text: str) -> str:
        # TODO: implement paraphrase generation via LLM
        return text
