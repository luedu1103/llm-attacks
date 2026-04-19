from typing import Any
from attacks.base import Attack


class ParaphraseAttack(Attack):
    """Paraphrase sentences while preserving meaning.

    Placeholder — returns example unchanged until implemented.
    """

    def apply(self, example: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement paraphrase generation via LLM
        return dict(example)
