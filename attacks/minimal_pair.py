from typing import Any
from attacks.base import Attack


class MinimalPairAttack(Attack):
    """Generate minimal pairs that change label-relevant features.

    Placeholder — returns example unchanged until implemented.
    """

    def apply(self, example: dict[str, Any]) -> dict[str, Any]:
        # TODO: implement minimal-pair generation
        return dict(example)
