import json
import re

from attacks.base import Attack
from utils.llm_client import chat
from utils.text_utils import preserve_case


class SynonymAttack(Attack):
    """Replace content words with synonyms using a single context-aware LLM call.

    The LLM receives the full sentence and selects which words to replace,
    avoiding word-sense ambiguity that arises when words are sent in isolation.
    """

    def _perturb_text(self, text: str) -> str:
        # \w{3,} pre-filters tokens too short to have meaningful synonyms;
        # the LLM further narrows to content words via the prompt.
        words = re.findall(r"\b\w{3,}\b", text, flags=re.UNICODE)
        n_replace = max(0, round(len(words) * self.intensity))
        if n_replace == 0:
            return text

        replacements = self._get_replacements(text, n_replace)
        return self._apply_replacements(text, replacements)

    def _get_replacements(self, text: str, n: int) -> dict[str, str]:
        """Single LLM call with full sentence context.

        Sending the whole sentence (rather than isolated words) lets the model
        resolve polysemy — e.g. "banco" → "entidad" vs "asiento" — before
        choosing a synonym.
        """
        prompt = (
            f"Tarea: en el siguiente texto en español, selecciona exactamente {n} "
            f"palabra(s) de contenido (sustantivos, verbos de acción, adjetivos o adverbios) "
            f"y proporciona un sinónimo para cada una.\n\n"
            f"Reglas estrictas:\n"
            f"- Usa el contexto completo de la oración para elegir el sinónimo correcto.\n"
            f"- No selecciones artículos, preposiciones, conjunciones, pronombres ni verbos auxiliares.\n"
            f"- El sinónimo debe encajar gramaticalmente en la misma posición.\n"
            f"- NO inventes palabras; usa únicamente español real.\n"
            f"- Si no hay suficientes palabras de contenido, devuelve menos reemplazos.\n"
            f"- Responde SOLO con JSON, sin texto adicional.\n"
            f'- Formato exacto: {{"replacements": [{{"original": "palabra", "synonym": "sinónimo"}}, ...]}}\n\n'
            f'Texto: "{text}"\n\n'
            f"JSON:"
        )
        return self._parse_response(chat(prompt).strip())

    def _parse_response(self, response: str) -> dict[str, str]:
        """Parse LLM output into an original→synonym mapping.

        The regex fallback handles models that wrap JSON in markdown fences or
        prepend prose before the object.
        """
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response, flags=re.DOTALL).strip()
        try:
            entries = json.loads(cleaned).get("replacements", [])
        except (json.JSONDecodeError, AttributeError):
            entries = [
                {"original": m.group(1), "synonym": m.group(2)}
                for m in re.finditer(
                    r'"original"\s*:\s*"([^"]+)".*?"synonym"\s*:\s*"([^"]+)"', cleaned
                )
            ]

        result: dict[str, str] = {}
        for entry in entries:
            orig = entry.get("original", "").strip().lower()
            syn = entry.get("synonym", "").strip().lower()
            if orig and syn and orig != syn and re.search(r"\w", syn):
                result[orig] = syn
        return result

    def _apply_replacements(self, text: str, replacements: dict[str, str]) -> str:
        # \b word boundaries prevent partial-word matches (e.g. "el" inside "el elemento").
        for orig, syn in replacements.items():
            pattern = re.compile(rf"\b{re.escape(orig)}\b", re.IGNORECASE | re.UNICODE)
            text = pattern.sub(lambda m: preserve_case(m.group(0), syn), text)
        return text
