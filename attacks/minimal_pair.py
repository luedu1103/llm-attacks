import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from attacks.base import Attack


class PerturbationType(Enum):
    NEGATION       = "negation"
    QUANTIFIER     = "quantifier"
    TEMPORAL       = "temporal"
    MODAL          = "modal"
    POLARITY       = "polarity"
    NONE           = "none"


@dataclass
class PerturbationResult:
    original: str
    perturbed: str
    changed_word: Optional[str]
    replacement: Optional[str]
    perturbation_type: PerturbationType
    was_modified: bool


SUBSTITUTION_RULES: list[tuple[str, str, PerturbationType]] = [

    # ── Negaciones directas ──────────────────────────────────────────────────
    # "no" inicial o aislado → eliminarlo (caso especial, ver lógica abajo)
    (r'\bno\b',          '',           PerturbationType.NEGATION),
    (r'\bnunca\b',       'siempre',    PerturbationType.NEGATION),
    (r'\bsiempre\b',     'nunca',      PerturbationType.NEGATION),
    (r'\bjamás\b',       'siempre',    PerturbationType.NEGATION),
    (r'\bnadie\b',       'alguien',    PerturbationType.NEGATION),
    (r'\balguien\b',     'nadie',      PerturbationType.NEGATION),
    (r'\bnada\b',        'algo',       PerturbationType.NEGATION),
    (r'\balgo\b',        'nada',       PerturbationType.NEGATION),
    (r'\bningún\b',      'algún',      PerturbationType.NEGATION),
    (r'\bninguna\b',     'alguna',     PerturbationType.NEGATION),
    (r'\bninguno\b',     'alguno',     PerturbationType.NEGATION),
    (r'\bningunos\b',    'algunos',    PerturbationType.NEGATION),
    (r'\bningunas\b',    'algunas',    PerturbationType.NEGATION),
    (r'\balgún\b',       'ningún',     PerturbationType.NEGATION),
    (r'\balguna\b',      'ninguna',    PerturbationType.NEGATION),
    (r'\balguno\b',      'ninguno',    PerturbationType.NEGATION),
    (r'\balgunos\b',     'ninguno',    PerturbationType.NEGATION),
    (r'\balgunas\b',     'ningunas',   PerturbationType.NEGATION),

    # ── Cuantificadores ──────────────────────────────────────────────────────
    (r'\btodo\b',        'ningún',     PerturbationType.QUANTIFIER),
    (r'\btoda\b',        'ninguna',    PerturbationType.QUANTIFIER),
    (r'\btodos\b',       'ninguno',    PerturbationType.QUANTIFIER),
    (r'\btodas\b',       'ninguna',    PerturbationType.QUANTIFIER),
    (r'\bmucho\b',       'poco',       PerturbationType.QUANTIFIER),
    (r'\bmucha\b',       'poca',       PerturbationType.QUANTIFIER),
    (r'\bmuchos\b',      'pocos',      PerturbationType.QUANTIFIER),
    (r'\bmuchas\b',      'pocas',      PerturbationType.QUANTIFIER),
    (r'\bpoco\b',        'mucho',      PerturbationType.QUANTIFIER),
    (r'\bpoca\b',        'mucha',      PerturbationType.QUANTIFIER),
    (r'\bpocos\b',       'muchos',     PerturbationType.QUANTIFIER),
    (r'\bpocas\b',       'muchas',     PerturbationType.QUANTIFIER),
    (r'\bsolo\b',        'también',    PerturbationType.QUANTIFIER),
    (r'\bsólo\b',        'también',    PerturbationType.QUANTIFIER),
    (r'\btambién\b',     'tampoco',    PerturbationType.QUANTIFIER),
    (r'\btampoco\b',     'también',    PerturbationType.QUANTIFIER),
    (r'\bambos\b',       'ninguno',    PerturbationType.QUANTIFIER),
    (r'\bambas\b',       'ninguna',    PerturbationType.QUANTIFIER),

    # ── Conectores temporales ────────────────────────────────────────────────
    (r'\bantes\b',       'después',    PerturbationType.TEMPORAL),
    (r'\bdespués\b',     'antes',      PerturbationType.TEMPORAL),
    (r'\btodavía\b',     'ya',         PerturbationType.TEMPORAL),
    (r'\baún\b',         'ya',         PerturbationType.TEMPORAL),
    (r'\bya\b',          'todavía',    PerturbationType.TEMPORAL),
    (r'\brecientemente\b', 'antiguamente', PerturbationType.TEMPORAL),
    (r'\bactualmente\b', 'anteriormente', PerturbationType.TEMPORAL),
    (r'\bpronto\b',      'tarde',      PerturbationType.TEMPORAL),
    (r'\btarde\b',       'pronto',     PerturbationType.TEMPORAL),
    (r'\bprimero\b',     'último',     PerturbationType.TEMPORAL),
    (r'\búltimo\b',      'primero',    PerturbationType.TEMPORAL),

    # ── Verbos modales ───────────────────────────────────────────────────────
    (r'\bpuede\b',       'no puede',   PerturbationType.MODAL),
    (r'\bpueden\b',      'no pueden',  PerturbationType.MODAL),
    (r'\bdebe\b',        'no debe',    PerturbationType.MODAL),
    (r'\bdeben\b',       'no deben',   PerturbationType.MODAL),
    (r'\bpodría\b',      'no podría',  PerturbationType.MODAL),
    (r'\bpodrían\b',     'no podrían', PerturbationType.MODAL),
    (r'\bdebería\b',     'no debería', PerturbationType.MODAL),
    (r'\bdeberían\b',    'no deberían',PerturbationType.MODAL),

    # ── Polaridad / valencia ─────────────────────────────────────────────────
    (r'\bpositivo\b',    'negativo',   PerturbationType.POLARITY),
    (r'\bnegativo\b',    'positivo',   PerturbationType.POLARITY),
    (r'\bfavorable\b',   'desfavorable', PerturbationType.POLARITY),
    (r'\bdesfavorable\b','favorable',  PerturbationType.POLARITY),
    (r'\bverdadero\b',   'falso',      PerturbationType.POLARITY),
    (r'\bfalso\b',       'verdadero',  PerturbationType.POLARITY),
    (r'\bcorrecto\b',    'incorrecto', PerturbationType.POLARITY),
    (r'\bincorrecto\b',  'correcto',   PerturbationType.POLARITY),
    (r'\bposible\b',     'imposible',  PerturbationType.POLARITY),
    (r'\bimposible\b',   'posible',    PerturbationType.POLARITY),
]


def _normalize(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def _match_case(original: str, replacement: str) -> str:
    if not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def _apply_no_removal(text: str) -> Optional[str]:
    """
    Caso especial: eliminar 'no' y limpiar el espacio sobrante.
    Evita doble espacio o 'no' al final de oración.
    """
    pattern = r'\bno\s+'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        result = text[:match.start()] + text[match.end():]
        if match.start() == 0 and result:
            result = result[0].upper() + result[1:]
        return result
    return None


class MinimalPairAttack(Attack):
    """
    Ataque de par mínimo robusto basado en reglas lingüísticas.

    Estrategia:
    1. Recorre las reglas en orden de prioridad.
    2. Aplica el PRIMER match encontrado (solo una sustitución).
    3. Preserva capitalización y estructura del texto original.
    4. Devuelve un PerturbationResult con metadata para trazabilidad.
    """

    def _perturb_text(self, text: str) -> str:
        result = self._apply_rules(text)
        return result.perturbed

    def _apply_rules(self, text: str) -> PerturbationResult:
        normalized = _normalize(text)
        
        max_substitutions = max(1, round(self.intensity * len(SUBSTITUTION_RULES)))
        substitutions_done = 0
        
        current_text = text
        last_result = PerturbationResult(
            original=text,
            perturbed=text,
            changed_word=None,
            replacement=None,
            perturbation_type=PerturbationType.NONE,
            was_modified=False,
        )

        for pattern, replacement, p_type in SUBSTITUTION_RULES:
            if substitutions_done >= max_substitutions:
                break
                
            flags = re.IGNORECASE | re.UNICODE
            normalized_current = _normalize(current_text)
            match = re.search(pattern, normalized_current, flags=flags)
            if not match:
                continue

            original_word = match.group()

            if pattern == r'\bno\b' and replacement == '':
                perturbed = _apply_no_removal(current_text)
                if perturbed is None:
                    continue
                current_text = perturbed
            else:
                final_replacement = _match_case(original_word, replacement)
                current_text = re.sub(
                    pattern, final_replacement, current_text, count=1, flags=flags
                )

            last_result = PerturbationResult(
                original=text,
                perturbed=current_text,
                changed_word=original_word,
                replacement=replacement or '∅',
                perturbation_type=p_type,
                was_modified=True,
            )
            substitutions_done += 1

        return last_result