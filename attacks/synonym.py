from attacks.base import Attack
import random

from utils.text_utils import (
    detokenize,
    n_modifications,
    preserve_case,
    sample_indices,
    simple_tokenize,
    word_indices,
)

_SIMPLE_SYNONYMS: dict[str, list[str]] = {
    "llegó": ["arribó", "vino"],
    "temprano": ["pronto", "anticipadamente"],
    "encontró": ["halló", "localizó"],
    "nadie": ["ninguno", "ninguna"],
    "sala": ["habitación", "salón"],
    "decidió": ["resolvió", "optó"],
    "esperar": ["aguardar"],
    "silencio": ["callada", "quietud"],
    "informe": ["reporte", "documento"],
    "equipo": ["grupo"],
    "revisará": ["verificará", "comprobará"],
    "datos": ["información"],
    "publicar": ["difundir", "divulgar"],
    "conclusiones": ["resultados"],
    "experimento": ["prueba"],
    "repetible": ["reproducible"],
    "protocolo": ["procedimiento"],
    "documentado": ["registrado"],
    "cualquiera": ["cualquier"],
    "seguir": ["continuar"],
    "pasos": ["etapas"],
    "rutina": ["hábito"],
    "constante": ["estable"],
    "mejorar": ["optimizar"],
    "comprensión": ["entendimiento"],
    "quizá": ["tal vez"],
    "tren": ["ferrocarril"],
    "retrace": ["se demore"],
    "salimos": ["partimos"],
    "tiempo": ["margen"],
    "evitar": ["prevenir"],
    "problemas": ["inconvenientes"],
    "nunca": ["jamás"],
    "contraseña": ["clave"],
    "dato": ["información"],
    "sensible": ["delicado"],
    "comprometer": ["poner en riesgo"],
    "seguridad": ["protección"],
    "docente": ["profesor"],
    "explicó": ["aclaró", "expuso"],
    "tema": ["asunto"],
    "propuso": ["planteó"],
    "ejercicios": ["prácticas"],
    "estudiantes": ["alumnos"],
    "practicaron": ["ensayaron"],
    "grupo": ["equipo"],
    "jamás": ["nunca"],
    "aceptó": ["admitió", "consintió"],
    "trato": ["acuerdo"],
    "prefirió": ["eligió"],
    "perder": ["ceder"],
    "dinero": ["capital"],
    "principios": ["valores"],
    "hipótesis": ["suposición"],
    "correcta": ["válida"],
    "faltan": ["carecen"],
    "evidencias": ["pruebas"],
    "comité": ["comisión"],
    "pide": ["solicita"],
    "pruebas": ["evidencias"],
    "sistema": ["mecanismo"],
    "falló": ["fracasó"],
    "respaldo": ["copia"],
    "copias": ["réplicas"],
    "recuperación": ["restauración"],
    "imposible": ["inviable"],
}


def _wordnet_synonym(word: str) -> str | None:
    """Try to get a Spanish synonym from NLTK WordNet/OMW if available.

    This is best-effort and will silently fall back.
    """
    try:
        from nltk.corpus import wordnet as wn  # type: ignore

        synsets = wn.synsets(word, lang="spa")
        for s in synsets:
            for lemma in s.lemma_names(lang="spa"):
                lemma = lemma.replace("_", " ")
                if lemma.lower() != word.lower():
                    # Return a single-token synonym only
                    if " " not in lemma:
                        return lemma
        return None
    except Exception:
        return None


class SynonymAttack(Attack):
    """Lexical substitution: replace a fraction of words with near-synonyms.

    Lookup priority: static dictionary (_SIMPLE_SYNONYMS) → NLTK WordNet/OMW.
    WordNet is optional; the attack degrades gracefully if it is not installed.
    Case of the source token is transferred to the replacement via preserve_case.
    """

    def _perturb_text(self, text: str) -> str:
        """Replace round(len(words) * intensity) words with synonyms, chosen at random."""
        tokens = simple_tokenize(text)
        if not tokens:
            return text

        widx = word_indices(tokens)
        if not widx:
            return text

        # Resolve replacements eagerly so each token is evaluated only once.
        candidate_map: dict[int, str] = {}
        for i in widx:
            lw = tokens[i].lower()
            if lw in _SIMPLE_SYNONYMS:
                candidate_map[i] = random.choice(_SIMPLE_SYNONYMS[lw])
            else:
                wn_repl = _wordnet_synonym(lw)
                if wn_repl:
                    candidate_map[i] = wn_repl

        n = min(len(candidate_map), n_modifications(len(widx), self.intensity))
        chosen = set(sample_indices(list(candidate_map), n, rng=random))

        new_tokens = list(tokens)
        for i in chosen:
            new_tokens[i] = preserve_case(tokens[i], candidate_map[i])

        return detokenize(new_tokens)
