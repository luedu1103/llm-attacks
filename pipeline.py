"""Main pipeline: dataset → attack → evaluation."""

from __future__ import annotations

import argparse
import json
import logging
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

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "synonym": SynonymAttack,
    "paraphrase": ParaphraseAttack,
    "minimal_pair": MinimalPairAttack,
    "shortcut_removal": ShortcutRemovalAttack,
}

_LLM_ATTACKS = {"synonym", "paraphrase"}

CHUNK_SIZE = 50


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_dataset(dataset: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def _run_sequential(
    attack: Attack,
    remaining: list[dict],
    perturbed: list[dict],
    checkpoint_path: Path,
    pbar: tqdm,
    already_done: int,
) -> None:
    """Process examples one by one. Used for regex-based attacks."""
    SAVE_EVERY = 10
    for i, ex in enumerate(remaining):
        try:
            perturbed.append(attack.apply(ex))
        except Exception as e:
            logger.warning(
                "apply() failed on example %d, keeping original. Error: %s",
                already_done + i,
                e,
            )
            perturbed.append(ex)
        pbar.update(1)

        if (i + 1) % SAVE_EVERY == 0:
            save_dataset(perturbed, checkpoint_path)


def _run_parallel(
    attack: Attack,
    remaining: list[dict],
    perturbed: list[dict],
    checkpoint_path: Path,
    pbar: tqdm,
    already_done: int,
) -> None:
    """Process examples in parallel chunks. Used for LLM-based attacks."""
    for chunk_start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_start : chunk_start + CHUNK_SIZE]
        try:
            results = attack.apply_dataset(chunk)
        except Exception as e:
            logger.warning(
                "apply_dataset() failed on chunk starting at %d, keeping originals. Error: %s",
                already_done + chunk_start,
                e,
            )
            results = chunk

        perturbed.extend(results)
        pbar.update(len(chunk))
        save_dataset(perturbed, checkpoint_path)


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

    use_parallel = attack_name in _LLM_ATTACKS
    mode = "parallel" if use_parallel else "sequential"
    print(f"Applying attack '{attack_name}' (intensity={intensity}, mode={mode})")

    with tqdm(total=len(dataset), initial=already_done, unit="ex") as pbar:
        if use_parallel:
            _run_parallel(
                attack, remaining, perturbed, checkpoint_path, pbar, already_done
            )
        else:
            _run_sequential(
                attack, remaining, perturbed, checkpoint_path, pbar, already_done
            )

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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args.attack, args.intensity, args.dataset, args.output)


if __name__ == "__main__":
    main()
