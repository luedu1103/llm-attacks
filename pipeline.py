"""Main pipeline: dataset → attack → evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
from attacks import SynonymAttack, ParaphraseAttack, MinimalPairAttack, ShortcutRemovalAttack
from attacks.base import Attack

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "synonym": SynonymAttack,
    "paraphrase": ParaphraseAttack,
    "minimal_pair": MinimalPairAttack,
    "shortcut_removal": ShortcutRemovalAttack,
}


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_dataset(dataset: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def run(attack_name: str, intensity: float, dataset_path: Path, output_path: Path | None) -> list[dict]:
    if attack_name not in ATTACK_REGISTRY:
        print(f"Unknown attack '{attack_name}'. Available: {list(ATTACK_REGISTRY)}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"  {len(dataset)} examples loaded")

    attack_cls = ATTACK_REGISTRY[attack_name]
    attack = attack_cls(intensity=intensity)
    print(f"Applying attack '{attack_name}' (intensity={intensity})")
    perturbed = attack.apply_dataset(dataset)

    if output_path is None:
        output_path = config.PERTURBED_DATA_DIR / f"{dataset_path.stem}_{attack_name}_{intensity}.json"

    save_dataset(perturbed, output_path)
    print(f"Perturbed dataset saved to {output_path}")
    return perturbed


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Benchmark pipeline")
    parser.add_argument("--attack", required=True, choices=list(ATTACK_REGISTRY), help="Attack type")
    parser.add_argument("--intensity", type=float, default=config.DEFAULT_INTENSITY, help="Attack intensity [0, 1]")
    parser.add_argument("--dataset", type=Path, default=config.DEFAULT_DATASET, help="Path to input dataset JSON")
    parser.add_argument("--output", type=Path, default=None, help="Path to save perturbed dataset JSON")
    args = parser.parse_args()

    run(args.attack, args.intensity, args.dataset, args.output)


if __name__ == "__main__":
    main()
