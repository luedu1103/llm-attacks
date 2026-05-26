"""
visualize.py — Generador de visualizaciones para los resultados de evaluación.

Lee el archivo eval_results.json y genera gráficos comparativos.

Uso:
    python -m evaluation.visualize --results evaluation/results/full/eval_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(results_path: Path, output_dir: Path) -> None:
    if not results_path.exists():
        print(f"Error: No se encontró el archivo de resultados en {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    orig_acc = data["original"]["accuracy"]
    perturbations = data.get("perturbations", [])

    if not perturbations:
        print("No hay datos de perturbaciones para visualizar.")
        return

    df = pd.DataFrame(perturbations)
    
    # Rellenar valores nulos de 'intensity' con un valor string para la leyenda
    if "intensity" in df.columns:
        df["intensity"] = df["intensity"].fillna("N/A")

    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Gráfico de Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="attack", y="accuracy", hue="intensity", palette="Blues_d")
    plt.axhline(orig_acc, color="red", linestyle="--", label=f"Original Accuracy ({orig_acc:.2f})")
    plt.title("Precisión del Modelo: Original vs Ataques")
    plt.ylabel("Accuracy")
    plt.xlabel("Tipo de Ataque")
    plt.legend(title="Intensidad")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
    plt.close()

    # Gráfico de Flip Rate
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="attack", y="flip_rate", hue="intensity", palette="Oranges_d")
    plt.title("Tasa de Cambio de Respuesta (Flip Rate) por Ataque")
    plt.ylabel("Flip Rate (Tasa de error inducida)")
    plt.xlabel("Tipo de Ataque")
    plt.legend(title="Intensidad")
    plt.tight_layout()
    plt.savefig(output_dir / "flip_rate.png", dpi=300)
    plt.close()

    # Visualizaciones: Curvas de Sensibilidad
    df_numeric = df[df["intensity"] != "N/A"].copy() # Filtro de valores para graficar de forma numérica
    if not df_numeric.empty:
        df_numeric["intensity"] = pd.to_numeric(df_numeric["intensity"])
        
        # "Baseline" (intensidad 0.0) para que la curva empiece en el original
        baseline_rows = []
        for atk in df_numeric["attack"].unique():
            baseline_rows.append({"attack": atk, "intensity": 0.0, "accuracy": orig_acc, "delta_accuracy": 0.0, "flip_rate": 0.0})
        
        df_plot = pd.concat([pd.DataFrame(baseline_rows), df_numeric], ignore_index=True)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_plot, x="intensity", y="accuracy", hue="attack", marker="o", linewidth=2.5, markersize=8)
        plt.title("Curvas de Sensibilidad: Precisión vs Intensidad")
        plt.ylabel("Accuracy")
        plt.xlabel("Intensidad de la Perturbación")
        plt.legend(title="Ataque")
        plt.tight_layout()
        plt.savefig(output_dir / "curva_sensibilidad.png", dpi=300)
        plt.close()

    print(f"Visualizaciones generadas exitosamente en: {output_dir}")
    print(f"  - {output_dir / 'accuracy_comparison.png'}")
    print(f"  - {output_dir / 'flip_rate.png'}")
    if not df_numeric.empty:
        print(f"  - {output_dir / 'curva_sensibilidad.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar gráficas de resultados")
    parser.add_argument("--results", type=Path, required=True,
                        help="Ruta al archivo eval_results.json")
    parser.add_argument("--outdir", type=Path, default=Path("evaluation/results/plots"),
                        help="Directorio donde guardar las imágenes")
    args = parser.parse_args()
    plot_results(args.results, args.outdir)