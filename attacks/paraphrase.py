from attacks.base import Attack
from utils.llm_client import chat


class ParaphraseAttack(Attack):
    """Rewrite sentences preserving meaning using an LLM."""

    def _perturb_text(self, text: str) -> str:
        if self.intensity < 0.3:
            level = "leve: cambia solo el orden de algunas palabras o reemplaza 1-2 frases"
        elif self.intensity < 0.5:
            level = "moderado: reescribe algunas oraciones con estructura diferente"
        else:
            level = "profundo: reescribe completamente la estructura de las oraciones"

        prompt = (
            f"Tarea: parafrasear un texto en español con nivel de cambio {level}.\n"
            f"Reglas estrictas:\n"
            f"- El significado debe ser IDÉNTICO al original.\n"
            f"- NO agregues información nueva ni expandas el texto.\n"
            f"- NO elimines información existente.\n"
            f"- NO completes frases incompletas.\n"
            f"- Si el texto es una frase incompleta, devuélvela igualmente incompleta.\n"
            f"- Conserva los números, nombres propios y términos técnicos.\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Después de estudiar, aprobó el examen.\"\n"
            f"Salida: \"Aprobó el examen tras haber estudiado.\"\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Se puede inferir que\"\n"
            f"Salida: \"Es posible concluir que\"\n\n"
            f"Ahora aplica la tarea al siguiente texto y devuelve ÚNICAMENTE el resultado:\n"
            f"{text}"
        )
        return chat(prompt)
