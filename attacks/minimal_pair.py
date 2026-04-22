from attacks.base import Attack
from utils.llm_client import chat


class MinimalPairAttack(Attack):
    """Change one critical word to alter the meaning minimally using an LLM."""

    def _perturb_text(self, text: str) -> str:
        prompt = (
            f"Tarea: modificar UN SOLO elemento crítico de un texto en español "
            f"para cambiar sutilmente su significado.\n"
            f"Reglas estrictas:\n"
            f"- Cambia ÚNICAMENTE una negación, cuantificador o conector temporal.\n"
            f"- El resto del texto debe quedar EXACTAMENTE igual, palabra por palabra.\n"
            f"- NO agregues palabras ni contenido nuevo.\n"
            f"- NO elimines palabras (salvo la que cambias).\n"
            f"- NO completes frases incompletas.\n"
            f"- Si no hay elemento crítico que cambiar, devuelve el texto sin modificar.\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Juan no fue al mercado.\"\n"
            f"Salida: \"Juan fue al mercado.\"\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Siempre llega tarde a las reuniones.\"\n"
            f"Salida: \"Nunca llega tarde a las reuniones.\"\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Se puede inferir que\"\n"
            f"Salida: \"Se puede inferir que\"\n\n"
            f"Ahora aplica la tarea al siguiente texto y devuelve ÚNICAMENTE el resultado:\n"
            f"{text}"
        )
        return chat(prompt)
