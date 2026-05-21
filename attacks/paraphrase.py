from attacks.base import Attack
from utils.llm_client import chat


class ParaphraseAttack(Attack):
    """Rewrite sentences preserving meaning using an LLM."""

    SIMILARITY_THRESHOLD = 0.50

    def _perturb_text(self, text: str) -> str:
        if self.intensity < 0.3:
            level = "leve"
        elif self.intensity < 0.5:
            level = "moderado"
        else:
            level = "profundo"

        lines = text.split("\n", 1)
        if len(lines) == 2 and not lines[0].strip().startswith(
            ("I.", "II.", "1.", "-")
        ):
            title = lines[0]
            body = lines[1]
        else:
            title = None
            body = text

        prompt = (
            f"Parafrasea este texto en español (nivel {level}). "
            f"Mismo significado, sin agregar ni eliminar información, conserva números y nombres propios. "
            f"Si está incompleto, devuélvelo incompleto. Solo devuelve el resultado:\n{body}"
        )

        result = chat(prompt)

        if title is not None:
            result = f"{title}\n{result}"

        return result
