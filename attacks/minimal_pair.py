from typing import Any
from attacks.base import Attack


class MinimalPairAttack(Attack):
    """Generate minimal pairs that change label-relevant features.

    Placeholder — returns example unchanged until implemented.
    """

    def _perturb_text(self, text: str) -> str:
        # TODO: implement minimal-pair generation
        return text
