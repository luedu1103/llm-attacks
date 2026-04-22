from attacks.base import Attack
from utils.llm_client import chat


class SynonymAttack(Attack):
    """Replace words with synonyms using an LLM."""

    def _perturb_text(self, text: str) -> str:
        pct = int(self.intensity * 100)
        prompt = (
            f"Tarea: reemplazar aproximadamente el {pct}% de las palabras de un texto "
            f"en español por sinónimos equivalentes.\n"
            f"Reglas estrictas:\n"
            f"- Sustituye SOLO palabras por sinónimos de igual significado.\n"
            f"- NO agregues palabras ni contenido nuevo.\n"
            f"- NO elimines palabras existentes.\n"
            f"- NO completes frases incompletas.\n"
            f"- NO corrijas errores del texto original.\n"
            f"- Conserva la puntuación, los números y el formato original.\n"
            f"- Si el texto es una frase incompleta, devuélvela igualmente incompleta.\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"El niño estaba feliz por su regalo.\"\n"
            f"Salida: \"El niño estaba contento por su obsequio.\"\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Se puede inferir que\"\n"
            f"Salida: \"Se puede deducir que\"\n\n"
            f"Ahora aplica la tarea al siguiente texto y devuelve ÚNICAMENTE el resultado:\n"
            f"{text}"
        )
        return chat(prompt)
