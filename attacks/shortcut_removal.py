import re

from attacks.base import Attack


class ShortcutRemovalAttack(Attack):
    """
    Remove explicit logical/causal connectors from text.

    This attack targets linguistic "shortcuts" — connectors that signal
    logical or causal relationships — to test whether a model relies on
    surface-level cues rather than deeper semantic understanding.

    Intensity controls how aggressively connectors are removed:
      - Low  (< 0.4): only remove isolated single-word connectors
      - Mid  (0.4–0.7): also remove short multi-word phrases (2 words)
      - High (> 0.7): remove all connectors including long phrases
    """

    FIELDS_TO_PERTURB: list[str] = ["question"]
    SIMILARITY_THRESHOLD = 0.85

    # Connectors grouped by token length for intensity-based filtering
    _SHORTCUTS_BY_LENGTH: dict[int, list[str]] = {
        1: [
            "porque",
            "pues",
            "como",
            "entonces",
            "luego",
            "aunque",
            "además",
            "incluso",
            "también",
        ],
        2: [
            "dado que",
            "ya que",
            "puesto que",
            "visto que",
            "así que",
            "por tanto",
            "por eso",
            "por ello",
            "de modo que",
            "de ahí que",
            "sin embargo",
            "no obstante",
            "aun así",
            "si bien",
            "con todo",
            "a pesar de",
            "pese a que",
            "no sólo",
            "es más",
            "y así",
            "no solo",
            "después de",
            "antes de",
            "en cuanto",
            "para que",
            "a fin de",
            "debido a",
            "a causa de",
            "por consiguiente",
            "por lo tanto",
        ],
        3: [
            "en consecuencia",
            "de manera que",
            "a pesar de que",
            "por añadidura",
            "asimismo",
            "una vez que",
            "a medida que",
            "tan pronto como",
            "siempre que",
            "con tal de que",
            "a menos que",
            "en caso de que",
            "a condición de que",
            "con el objetivo de",
            "con miras a",
            "en un principio",
            "de otro lado",
        ],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pattern = self._build_pattern()

    def _build_pattern(self) -> re.Pattern:
        """Build a compiled regex from shortcuts filtered by intensity."""
        max_length = self._intensity_to_max_length()

        selected = [
            phrase
            for length, phrases in self._SHORTCUTS_BY_LENGTH.items()
            if length <= max_length
            for phrase in phrases
        ]

        selected.sort(key=len, reverse=True)
        escaped = [re.escape(s) for s in selected]
        return re.compile(
            r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)",
            re.IGNORECASE,
        )

    def _intensity_to_max_length(self) -> int:
        """Map intensity float to maximum connector token length."""
        if self.intensity < 0.4:
            return 1
        if self.intensity < 0.7:
            return 2
        return 3

    def _perturb_text(self, text: str) -> str:
        result = self._pattern.sub("", text)
        result = self._clean(result)
        return result

    @staticmethod
    def _clean(text: str) -> str:
        """Normalize whitespace and punctuation left by connector removal."""
        text = re.sub(r"\.\s*,", ".", text)
        text = re.sub(r"^[\s,;:]+", "", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\s([?.!,;:])", r"\1", text)

        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
