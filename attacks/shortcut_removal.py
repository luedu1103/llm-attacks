from attacks.base import Attack
from utils.llm_client import chat


class ShortcutRemovalAttack(Attack):
    """Remove explicit logical/causal connectors using an LLM."""

    def _perturb_text(self, text: str) -> str:
        prompt = (
            f"Tarea: eliminar conectores lógicos y causales de un texto en español.\n"
            f"Reglas estrictas:\n"
            f"- Elimina SOLO conectores como: porque, entonces, por tanto, sin embargo, "
            f"dado que, en consecuencia, luego, así que, por eso, no sólo...sino, "
            f"y así, en un principio, de otro lado, por ello, debido a, a causa de.\n"
            f"- NO agregues palabras ni contenido nuevo.\n"
            f"- NO corrijas errores del texto original.\n"
            f"- NO completes frases incompletas.\n"
            f"- Conserva exactamente todas las demás palabras.\n"
            f"- Si el texto no tiene conectores, devuélvelo sin modificar.\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Luego, la lava se enfrió y se convirtió en piedra.\"\n"
            f"Salida: \"La lava se enfrió y se convirtió en piedra.\"\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Juan llegó tarde porque perdió el bus.\"\n"
            f"Salida: \"Juan llegó tarde. Perdió el bus.\"\n\n"
            f"Ejemplo:\n"
            f"Entrada: \"Se puede inferir que\"\n"
            f"Salida: \"Se puede inferir que\"\n\n"
            f"Ahora aplica la tarea al siguiente texto y devuelve ÚNICAMENTE el resultado:\n"
            f"{text}"
        )
        return chat(prompt)
