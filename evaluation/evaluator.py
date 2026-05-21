"""evaluator.py — Hace que el LLM responda cada ítem del dataset y retorna predicciones.

Uso directo:
    from evaluation.evaluator import evaluate_dataset
    predictions = evaluate_dataset(dataset)   # lista de ints (índice de opción, base 0)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tqdm import tqdm

from utils.llm_client import chat

logger = logging.getLogger(__name__)

# Letras que corresponden a los índices 0-4
_OPTION_LETTERS = ["A", "B", "C", "D", "E"]

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
        # Intentar 1-based primero
        if 1 <= num <= n_options:
            return num - 1

    return None


# ──────────────────────────────────────────────
# Evaluación principal
# ──────────────────────────────────────────────

def evaluate_dataset(
    dataset: list[dict[str, Any]],
    desc: str = "Evaluando",
) -> list[int | None]:
    """Envía cada ítem al LLM y retorna la lista de predicciones.

    Retorna una lista de enteros (índice 0-based de la opción elegida).
    Si el LLM da una respuesta ininterpretable, guarda None para ese ítem.
    """
    predictions: list[int | None] = []

    for example in tqdm(dataset, desc=desc, unit="ítem"):
        prompt = _build_prompt(example)
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
                raw if "raw" in dir() else "N/A",
            )

        predictions.append(pred)

    return predictions
