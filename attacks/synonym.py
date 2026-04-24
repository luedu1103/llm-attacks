import random
import re

from attacks.base import Attack
from utils.llm_client import chat


class SynonymAttack(Attack):
    """Replace words with synonyms using an LLM by processing words in batches.

    Words are grouped into batches and sent to the LLM together, reducing
    the number of API calls while maintaining quality.
    """

    BATCH_SIZE = 10

    STOPWORDS = {
        # Articles
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        # Prepositions / conjunctions
        "de",
        "a",
        "en",
        "al",
        "del",
        "por",
        "con",
        "para",
        "pero",
        "o",
        "u",
        "y",
        "e",
        "ni",
        "si",
        "que",
        "como",
        "más",
        "más",
        # Pronouns
        "se",
        "lo",
        "le",
        "les",
        "me",
        "te",
        "nos",
        "os",
        "yo",
        "tú",
        "él",
        "ella",
        "ello",
        "nosotros",
        "vosotros",
        "ellos",
        "ellas",
        "su",
        "sus",
        "mi",
        "mis",
        "tu",
        "tus",
        "este",
        "esta",
        "estos",
        "estas",
        "ese",
        "esa",
        "esos",
        "esas",
        "aquel",
        "aquella",
        "aquellos",
        "aquellas",
        # Common verbs (ser/estar/haber/tener)
        "es",
        "son",
        "era",
        "eran",
        "fue",
        "fueron",
        "ser",
        "está",
        "están",
        "estaba",
        "estaban",
        "estuvo",
        "estuvieron",
        "hay",
        "había",
        "habían",
        "hubo",
        "hubieron",
        "tiene",
        "tienes",
        "tenemos",
        "tenéis",
        "tienen",
        "tenía",
        "tenías",
        "teníamos",
        "teníais",
        "tenían",
        "tendrá",
        "tendrás",
        "tendremos",
        "tendréis",
        "tendrán",
        # Common verbs (ver/decir/hablar/ir)
        "veo",
        "ves",
        "ve",
        "vemos",
        "veis",
        "ven",
        "dice",
        "dices",
        "decimos",
        "decís",
        "dicen",
        "dijo",
        "dije",
        "dijiste",
        "dijimos",
        "dijisteis",
        "dijeron",
        "hablo",
        "hablas",
        "habla",
        "hablamos",
        "habláis",
        "hablan",
        "voy",
        "vas",
        "va",
        "vamos",
        "vais",
        "van",
        # Negation / affirmation
        "no",
        "sí",
    }

    def _perturb_text(self, text: str) -> str:
        """Replace a percentage of content words with synonyms."""
        pct = int(self.intensity * 100)

        tokens = self._tokenize(text)

        candidates: list[int] = []
        for i, token in enumerate(tokens):
            core = re.sub(r"[^\w]", "", token, flags=re.UNICODE).lower()
            if not core or len(core) <= 2:
                continue
            if core in self.STOPWORDS:
                continue
            candidates.append(i)

        random.shuffle(candidates)
        n_replace = max(0, len(candidates) * pct // 100)
        if n_replace == 0:
            return text

        to_replace = candidates[:n_replace]

        words_to_process = [(i, tokens[i]) for i in to_replace]

        synonym_map: dict[str, str] = {}
        for batch_start in range(0, len(words_to_process), self.BATCH_SIZE):
            batch = words_to_process[batch_start : batch_start + self.BATCH_SIZE]
            unique_words = list(
                {re.sub(r"[^\w]", "", w, flags=re.UNICODE).lower() for _, w in batch}
            )
            batch_result = self._find_synonyms_batch(unique_words)
            synonym_map.update(batch_result)

        for i, original_token in words_to_process:
            core = re.sub(r"[^\w]", "", original_token, flags=re.UNICODE).lower()
            synonym = synonym_map.get(core)
            if synonym and synonym.lower() != core:
                tokens[i] = self._match_case(original_token, synonym)

        return "".join(tokens)

    def _tokenize(self, text: str) -> list[str]:
        """Split text into alternating word / non-word tokens, preserving spaces."""
        return re.findall(r"\w+|[^\w]+", text, flags=re.UNICODE)

    def _find_synonyms_batch(self, words: list[str]) -> dict[str, str]:
        """Ask the LLM for synonyms of a batch of words in one call.

        Returns a dict mapping each original word to its synonym (or itself
        if no good synonym was found).
        """
        word_list = "\n".join(f"- {w}" for w in words)
        prompt = (
            "Tarea: Para cada palabra de la lista, proporciona UN sinónimo en español.\n"
            "\n"
            "Reglas estrictas:\n"
            "- Responde SOLO con JSON, sin texto adicional, sin explicaciones.\n"
            '- El formato debe ser exactamente: {"palabra": "sinónimo", ...}\n'
            "- Si no hay un buen sinónimo, usa la misma palabra como valor.\n"
            "- NO inventes palabras; usa únicamente español real.\n"
            "- Un sinónimo por palabra, sin listas.\n"
            "\n"
            f"Palabras:\n{word_list}\n"
            "\n"
            "JSON:"
        )

        response = chat(prompt).strip()

        return self._parse_batch_response(response, words)

    def _parse_batch_response(
        self, response: str, original_words: list[str]
    ) -> dict[str, str]:
        """Parse the JSON response from the LLM into a word→synonym mapping."""
        import json

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", response, flags=re.DOTALL
        ).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {}
            for match in re.finditer(r'"([^"]+)"\s*:\s*"([^"]+)"', cleaned):
                data[match.group(1).lower()] = match.group(2)

        result: dict[str, str] = {}
        for word in original_words:
            synonym = data.get(word, word)
            if (
                not isinstance(synonym, str)
                or len(synonym) > 30
                or not re.search(r"\w", synonym)
            ):
                synonym = word
            result[word] = synonym.strip().lower()

        return result

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Apply the capitalisation style of *original* to *replacement*."""
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.capitalize()
        return replacement
