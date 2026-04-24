"""Main pipeline: dataset → attack → evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

import config
from attacks import (
    MinimalPairAttack,
    ParaphraseAttack,
    ShortcutRemovalAttack,
    SynonymAttack,
)
from attacks.base import Attack

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "synonym": SynonymAttack,
    "paraphrase": ParaphraseAttack,
    "minimal_pair": MinimalPairAttack,
    "shortcut_removal": ShortcutRemovalAttack,
}


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def run(
    attack_name: str, intensity: float, dataset_path: Path, output_path: Path | None
) -> list[dict]:
    if attack_name not in ATTACK_REGISTRY:
        print(
            f"Unknown attack '{attack_name}'. Available: {list(ATTACK_REGISTRY)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"  {len(dataset)} examples loaded")

    if output_path is None:
        output_path = (
            config.PERTURBED_DATA_DIR
            / f"{dataset_path.stem}_{attack_name}_{intensity}.json"
        )

    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    perturbed: list[dict] = []

    if checkpoint_path.exists():
        perturbed = load_dataset(checkpoint_path)
        print(f"  Resuming from checkpoint: {len(perturbed)} examples already done")

    already_done = len(perturbed)
    remaining = dataset[already_done:]

    attack_cls = ATTACK_REGISTRY[attack_name]
    attack = attack_cls(intensity=intensity)

    print(f"Applying attack '{attack_name}' (intensity={intensity})")

    SAVE_EVERY = 10

    with tqdm(total=len(dataset), initial=already_done, unit="ex") as pbar:
        for i, ex in enumerate(remaining):
            perturbed.append(attack.apply(ex))
            pbar.update(1)

            if (i + 1) % SAVE_EVERY == 0:
                save_dataset(perturbed, checkpoint_path)

    save_dataset(perturbed, output_path)
    print(f"Perturbed dataset saved to {output_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint removed")

    return perturbed


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Benchmark pipeline")
    parser.add_argument(
        "--attack", required=True, choices=list(ATTACK_REGISTRY), help="Attack type"
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=config.DEFAULT_INTENSITY,
        help="Attack intensity [0, 1]",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.DEFAULT_DATASET,
        help="Path to input dataset JSON",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to save perturbed dataset JSON"
    )
    args = parser.parse_args()

    run(args.attack, args.intensity, args.dataset, args.output)


if __name__ == "__main__":
    main()
