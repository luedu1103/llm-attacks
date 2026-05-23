"""run_eval.py — Orquestador principal de evaluación.

Evalúa el dataset original y todos los datasets perturbados disponibles,
calcula accuracy / delta_accuracy / flip_rate y guarda los resultados.

Uso:
    # Desde la raíz del proyecto:
    python -m evaluation.run_eval                        # usa sample por defecto
    python -m evaluation.run_eval --full                 # usa el dataset completo
    python -m evaluation.run_eval --original data/raw/sample_dataset.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import config
from evaluation.evaluator import evaluate_dataset
from evaluation.metrics import accuracy, delta_accuracy, flip_rate

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Helpers de I/O
# ──────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, ValueError):
            continue
    raise ValueError(f"No se pudo leer {path} con ningún encoding conocido")


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en: {path}")


def save_predictions(preds: list, path: Path) -> None:
    """Guarda predicciones para no tener que volver a llamar al LLM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f)


def load_predictions(path: Path) -> list | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────
# Lógica de evaluación
# ──────────────────────────────────────────────


def get_labels(dataset: list[dict]) -> list[int]:
    """Extrae las etiquetas correctas (campo 'answer') del dataset."""
    return [ex["answer"] for ex in dataset]


def run_evaluation(
    original_path: Path,
    perturbed_dir: Path,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 60)
    print("  EVALUACIÓN DE ROBUSTEZ LÉXICA")
    print("=" * 60)

    # ── 1. Evaluar dataset original ──────────────────────────────
    print(f"\n[1/2] Dataset original: {original_path.name}")

    pred_cache_path = output_dir / f"predictions_{original_path.stem}.json"

    # Verificar si ya está completo (mismo número de ítems que el dataset)
    original_dataset = load_dataset(original_path)
    cached = load_predictions(pred_cache_path)
    if cached is not None and len(cached) == len(original_dataset):
        print(f"  ✓ Predicciones completas en caché ({len(cached)} ítems)")
        original_preds = cached
    else:
        # evaluate_dataset retoma automáticamente desde el caché parcial
        original_preds = evaluate_dataset(
            original_dataset,
            desc="Original",
            cache_path=pred_cache_path,
        )

    labels = get_labels(original_dataset)

    # Filtrar ítems donde el LLM no pudo responder
    valid_mask = [p is not None for p in original_preds]
    n_invalid = valid_mask.count(False)
    if n_invalid > 0:
        print(f"  ⚠ {n_invalid} ítems con respuesta ininterpretable (se excluyen)")

    orig_preds_clean = [p for p, v in zip(original_preds, valid_mask) if v]
    labels_clean = [l for l, v in zip(labels, valid_mask) if v]

    orig_acc = accuracy(orig_preds_clean, labels_clean)
    print(f"  Accuracy original: {orig_acc:.4f} ({orig_acc * 100:.1f}%)")

    # ── 2. Evaluar cada dataset perturbado ───────────────────────
    print(f"\n[2/2] Datasets perturbados en: {perturbed_dir}")

    perturbed_files = sorted(perturbed_dir.glob("*.json"))
    if not perturbed_files:
        print("  ⚠ No se encontraron datasets perturbados.")
        return

    print(f"  Encontrados: {len(perturbed_files)} archivos\n")

    all_results = {
        "original": {
            "dataset": original_path.name,
            "accuracy": round(orig_acc, 4),
            "n_items": len(orig_preds_clean),
            "n_invalid": n_invalid,
        },
        "perturbations": [],
    }

    summary_rows = []

    for pert_path in perturbed_files:
        print(f"  → {pert_path.name}")

        pert_pred_cache = output_dir / f"predictions_{pert_path.stem}.json"
        pert_dataset = load_dataset(pert_path)

        # Alinear por id en caso de dataset perturbado incompleto
        pert_ids = [ex["id"] for ex in pert_dataset]
        pert_id_set = set(pert_ids)
        orig_id_to_idx = {ex["id"]: i for i, ex in enumerate(original_dataset)}

        if len(pert_dataset) != len(original_dataset):
            print(
                f"    ⚠ Dataset incompleto: {len(pert_dataset)}/{len(original_dataset)} ejemplos — se evalúan solo los presentes"
            )

        # Máscara combinada: válido en original Y presente en perturbado
        combined_mask = [
            v and (ex["id"] in pert_id_set)
            for ex, v in zip(original_dataset, valid_mask)
        ]

        cached_pert = load_predictions(pert_pred_cache)

        if cached_pert is not None and len(cached_pert) == len(pert_dataset):
            print(f"    ✓ Predicciones completas en caché ({len(cached_pert)} ítems)")
            pert_preds = cached_pert
        else:
            pert_preds = evaluate_dataset(
                pert_dataset,
                desc=f"  {pert_path.stem}",
                cache_path=pert_pred_cache,
            )

        # Mapear predicciones del perturbado por id
        pert_pred_by_id = {pert_ids[i]: pert_preds[i] for i in range(len(pert_ids))}

        # Construir listas alineadas usando combined_mask
        orig_preds_aligned = []
        pert_preds_aligned = []
        labels_aligned = []
        n_pert_invalid = 0

        for ex, orig_pred, label, keep in zip(
            original_dataset, original_preds, labels, combined_mask
        ):
            if not keep:
                continue
            pert_pred = pert_pred_by_id.get(ex["id"])
            if pert_pred is None:
                n_pert_invalid += 1
                continue
            orig_preds_aligned.append(orig_pred)
            pert_preds_aligned.append(pert_pred)
            labels_aligned.append(label)

        if n_pert_invalid > 0:
            print(
                f"    ⚠ {n_pert_invalid} ítems ininterpretables en perturbado (se excluyen de ambas listas)"
            )

        pert_acc = accuracy(pert_preds_aligned, labels_aligned)
        d_acc = delta_accuracy(orig_preds_aligned, pert_preds_aligned, labels_aligned)
        fr = flip_rate(orig_preds_aligned, pert_preds_aligned)

        # Parsear nombre del archivo para extraer ataque e intensidad
        # Formato esperado: {dataset_name}_{attack}_{intensity}.json
        # Usamos regex para encontrar la intensidad (float al final del stem)
        # y el ataque (token conocido justo antes de la intensidad)
        stem = pert_path.stem
        _KNOWN_ATTACKS = {"synonym", "paraphrase", "minimal_pair", "shortcut_removal"}
        m = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", stem)
        if m:
            intensity = float(m.group(1))
            prefix = stem[: m.start()]  # todo antes de la intensidad
            # Buscar el ataque conocido más largo que sea sufijo del prefix
            attack_name = next(
                (
                    a
                    for a in sorted(_KNOWN_ATTACKS, key=len, reverse=True)
                    if prefix.endswith(a) or prefix.endswith(a.replace("_", "_"))
                ),
                prefix.split("_")[-1],  # fallback: último token
            )
        else:
            intensity = None
            attack_name = stem

        result = {
            "file": pert_path.name,
            "attack": attack_name,
            "intensity": intensity,
            "accuracy": round(pert_acc, 4),
            "delta_accuracy": round(d_acc, 4),
            "flip_rate": round(fr, 4),
        }
        all_results["perturbations"].append(result)

        print(f"    accuracy={pert_acc:.4f}  Δacc={d_acc:+.4f}  flip_rate={fr:.4f}")
        summary_rows.append((attack_name, intensity, pert_acc, d_acc, fr))

    # ── 3. Guardar resultados JSON ────────────────────────────────
    results_path = output_dir / "eval_results.json"
    save_results(all_results, results_path)

    # ── 4. Imprimir tabla resumen ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Accuracy base (original): {orig_acc:.4f}\n")
    print(f"  {'Ataque':<22} {'Intens':>7} {'Acc':>7} {'ΔAcc':>8} {'FlipRate':>10}")
    print("  " + "-" * 58)
    for attack, intensity, acc, dacc, fr in sorted(
        summary_rows, key=lambda x: (x[0], x[1] or 0)
    ):
        intens_str = f"{intensity:.1f}" if intensity is not None else "?"
        print(f"  {attack:<22} {intens_str:>7} {acc:>7.4f} {dacc:>+8.4f} {fr:>10.4f}")
    print("=" * 60)
    print(f"\nResultados completos en: {results_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa robustez léxica: original vs perturbados"
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=config.RAW_DATA_DIR / "sample_dataset.json",
        help="Path al dataset original (default: sample_dataset.json)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Usar el dataset completo en vez del sample",
    )
    parser.add_argument(
        "--perturbed-dir",
        type=Path,
        default=None,
        help="Directorio con datasets perturbados (default: data/perturbed/sample o ready)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio donde guardar resultados (default: evaluation/results/sample o evaluation/results/full)",
    )
    parser.add_argument("--debug", action="store_true", help="Activa logs de debug")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolver paths según flags
    if args.full:
        original_path = config.DEFAULT_DATASET
        perturbed_dir = config.PERTURBED_DATA_DIR / "readyV1.5"
        output_dir = args.output_dir or (
            config.ROOT_DIR / "evaluation" / "results" / "full"
        )
    else:
        original_path = args.original
        perturbed_dir = args.perturbed_dir or (config.PERTURBED_DATA_DIR / "sample")
        output_dir = args.output_dir or (
            config.ROOT_DIR / "evaluation" / "results" / "sample"
        )

    run_evaluation(
        original_path=original_path,
        perturbed_dir=perturbed_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
