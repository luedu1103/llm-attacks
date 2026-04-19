from typing import Any
from attacks.base import Attack


class ShortcutRemovalAttack(Attack):
    """Remove lexical shortcuts (spurious correlations) from text.

    Placeholder — returns example unchanged until implemented.
    """

    def apply(self, example: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement shortcut detection and removal
        return dict(example)
