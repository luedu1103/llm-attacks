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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    original_preds = load_predictions(pred_cache_path)

    if original_preds is not None:
        print(f"  ✓ Predicciones en caché encontradas ({len(original_preds)} ítems)")
    else:
        original_dataset = load_dataset(original_path)
        original_preds = evaluate_dataset(original_dataset, desc="Original")
        save_predictions(original_preds, pred_cache_path)

    original_dataset = load_dataset(original_path)
    labels = get_labels(original_dataset)

    # Filtrar ítems donde el LLM no pudo responder
    valid_mask = [p is not None for p in original_preds]
    n_invalid = valid_mask.count(False)
    if n_invalid > 0:
        print(f"  ⚠ {n_invalid} ítems con respuesta ininterpretable (se excluyen)")

    orig_preds_clean = [p for p, v in zip(original_preds, valid_mask) if v]
    labels_clean = [l for l, v in zip(labels, valid_mask) if v]

    orig_acc = accuracy(orig_preds_clean, labels_clean)
    print(f"  Accuracy original: {orig_acc:.4f} ({orig_acc*100:.1f}%)")

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
        pert_preds = load_predictions(pert_pred_cache)

        if pert_preds is not None:
            print(f"    ✓ Predicciones en caché ({len(pert_preds)} ítems)")
        else:
            pert_dataset = load_dataset(pert_path)
            pert_preds = evaluate_dataset(pert_dataset, desc=f"  {pert_path.stem}")
            save_predictions(pert_preds, pert_pred_cache)

        # Alinear con los ítems válidos del original
        pert_preds_clean = [p for p, v in zip(pert_preds, valid_mask) if v]
        # Si el perturbado tiene algún None, reemplazar con la pred original
        # (conservador: no penalizar por fallo de parseo en perturbado)
        pert_preds_aligned = [
            pp if pp is not None else op
            for pp, op in zip(pert_preds_clean, orig_preds_clean)
        ]

        pert_acc = accuracy(pert_preds_aligned, labels_clean)
        d_acc = delta_accuracy(orig_preds_clean, pert_preds_aligned, labels_clean)
        fr = flip_rate(orig_preds_clean, pert_preds_aligned)

        # Parsear nombre del archivo para extraer ataque e intensidad
        # Formato esperado: dataset_{attack}_{intensity}.json
        stem = pert_path.stem  # ej: "dataset_synonym_0.3"
        parts = stem.split("_")
        # Buscar la intensidad (último token numérico)
        try:
            intensity = float(parts[-1])
            attack_name = "_".join(parts[1:-1])  # todo entre "dataset" e intensidad
        except ValueError:
            intensity = None
            attack_name = "_".join(parts[1:])

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
    for attack, intensity, acc, dacc, fr in sorted(summary_rows, key=lambda x: (x[0], x[1] or 0)):
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
        default=config.ROOT_DIR / "evaluation" / "results",
        help="Directorio donde guardar resultados y predicciones en caché",
    )
    parser.add_argument("--debug", action="store_true", help="Activa logs de debug")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolver paths según flags
    if args.full:
        original_path = config.DEFAULT_DATASET
        perturbed_dir = config.PERTURBED_DATA_DIR / "ready"
    else:
        original_path = args.original
        perturbed_dir = args.perturbed_dir or (config.PERTURBED_DATA_DIR / "sample")

    run_evaluation(
        original_path=original_path,
        perturbed_dir=perturbed_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
