"""evaluator.py — Hace que el LLM responda cada ítem del dataset y retorna predicciones.
 
Uso directo:
    from evaluation.evaluator import evaluate_dataset
    predictions = evaluate_dataset(dataset, cache_path=Path("evaluation/results/predictions_x.json"))
"""
 
from __future__ import annotations
 
import json
import logging
import re
from pathlib import Path
from typing import Any
 
from tqdm import tqdm
 
from utils.llm_client import chat
 
logger = logging.getLogger(__name__)
 
# Letras que corresponden a los índices 0-4
_OPTION_LETTERS = ["A", "B", "C", "D", "E"]
 
# Guardar caché cada N ítems
_SAVE_EVERY = 5
 
 
# ──────────────────────────────────────────────
# Helpers de caché parcial
# ──────────────────────────────────────────────
 
def _save_partial(predictions: list, cache_path: Path) -> None:
    """Guarda las predicciones acumuladas hasta ahora en el archivo de caché."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
 
 
def _load_partial(cache_path: Path) -> list:
    """Carga predicciones parciales si el archivo de caché existe."""
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return []
 
 
# ──────────────────────────────────────────────
# Construcción de prompts por tipo de tarea
# ──────────────────────────────────────────────
 
def _build_prompt(example: dict[str, Any]) -> str:
    """Construye el prompt correcto según el tipo de tarea."""
    task = example.get("task", "")
    question = example.get("question", "")
    options: list[str] = example.get("options", [])
 
    options_text = "\n".join(
        f"{_OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )
 
    if task == "sentence_ordering":
        prompt = (
            "Lee el siguiente ejercicio de ordenamiento de oraciones y elige "
            "la secuencia correcta.\n\n"
            f"{question}\n\n"
            f"Opciones:\n{options_text}\n\n"
            "Responde ÚNICAMENTE con la letra de la opción correcta (A, B, C, D o E)."
        )
 
    elif task == "sentence_elimination":
        context = example.get("context", "")
        context_block = f"Texto de referencia:\n{context}\n\n" if context else ""
        prompt = (
            "Lee el siguiente ejercicio y elige la opción que corresponde.\n\n"
            f"{context_block}"
            f"Pregunta: {question}\n\n"
            f"Opciones:\n{options_text}\n\n"
            "Responde ÚNICAMENTE con la letra de la opción correcta (A, B, C, D o E)."
        )
 
    else:
        # Fallback genérico para cualquier otra tarea (mcq, series, etc.)
        context = example.get("context", "")
        context_block = f"Contexto:\n{context}\n\n" if context else ""
        prompt = (
            f"{context_block}"
            f"Pregunta: {question}\n\n"
            f"Opciones:\n{options_text}\n\n"
            "Responde ÚNICAMENTE con la letra de la opción correcta (A, B, C, D o E)."
        )
 
    return prompt
 
 
# ──────────────────────────────────────────────
# Parseo de la respuesta del LLM
# ──────────────────────────────────────────────
 
def _parse_response(response: str, n_options: int) -> int | None:
    """Extrae el índice (0-based) de la opción elegida por el LLM.
 
    Acepta respuestas como: "A", "a)", "La respuesta es B", "2", etc.
    Devuelve None si no puede interpretarla.
    """
    response = response.strip().upper()
 
    # Intento 1: letra explícita (A-E)
    match = re.search(r"\b([A-E])\b", response)
    if match:
        letter = match.group(1)
        idx = _OPTION_LETTERS.index(letter)
        if idx < n_options:
            return idx
 
    # Intento 2: número (1-based o 0-based)
    match = re.search(r"\b([1-5])\b", response)
    if match:
        num = int(match.group(1))
        if 1 <= num <= n_options:
            return num - 1
 
    return None
 
 
# ──────────────────────────────────────────────
# Evaluación principal
# ──────────────────────────────────────────────
 
def evaluate_dataset(
    dataset: list[dict[str, Any]],
    desc: str = "Evaluando",
    cache_path: Path | None = None,
) -> list[int | None]:
    """Envía cada ítem al LLM y retorna la lista de predicciones.
 
    Si se proporciona cache_path:
    - Al iniciar, carga predicciones parciales previas y retoma desde ahí.
    - Cada _SAVE_EVERY ítems guarda el progreso automáticamente.
    - Al terminar, elimina el caché parcial (ya no hace falta).
 
    Retorna una lista de enteros (índice 0-based de la opción elegida).
    Si el LLM da una respuesta ininterpretable, guarda None para ese ítem.
    """
    # ── Retomar desde caché parcial si existe ──
    predictions: list[int | None] = []
    already_done = 0
 
    if cache_path is not None:
        predictions = _load_partial(cache_path)
        already_done = len(predictions)
        if already_done > 0:
            print(f"  ↺ Retomando desde ítem {already_done} (caché parcial encontrada)")
 
    remaining = dataset[already_done:]
 
    # ── Iterar sobre los ítems pendientes ──
    for i, example in enumerate(tqdm(remaining, desc=desc, unit="ítem", initial=already_done, total=len(dataset))):
        prompt = _build_prompt(example)
        raw = None
        try:
            raw = chat(prompt)
            pred = _parse_response(raw, n_options=len(example.get("options", [])))
        except Exception as e:
            logger.warning("Error al evaluar ítem %s: %s", example.get("id"), e)
            pred = None
 
        if pred is None:
            logger.warning(
                "No se pudo parsear respuesta para ítem %s. Respuesta cruda: %r",
                example.get("id"),
                raw,
            )
 
        predictions.append(pred)
 
        # ── Guardar caché parcial cada _SAVE_EVERY ítems ──
        if cache_path is not None and (i + 1) % _SAVE_EVERY == 0:
            _save_partial(predictions, cache_path)
            logger.debug("Caché parcial guardada: %d ítems", len(predictions))
 
    # ── Guardado final y limpieza ──
    if cache_path is not None:
        _save_partial(predictions, cache_path)
 
    return predictions