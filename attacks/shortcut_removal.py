from typing import Any
from attacks.base import Attack


class ShortcutRemovalAttack(Attack):
    """Remove lexical shortcuts (spurious correlations) from text.

    Placeholder — returns example unchanged until implemented.
    """

    def _perturb_text(self, text: str) -> str:
        # TODO: implement shortcut detection and removal
        return text
