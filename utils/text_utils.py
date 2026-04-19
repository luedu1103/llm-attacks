import random
import re
from typing import Iterable, Sequence


_WORD_RE = re.compile(r"^\w+$", flags=re.UNICODE)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def set_global_seed(seed: int) -> None:
    random.seed(seed)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def simple_tokenize(text: str) -> list[str]:
    """Very small tokenizer for Spanish-like text.

    Splits into word tokens and punctuation tokens.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def is_word_token(token: str) -> bool:
    return bool(_WORD_RE.match(token))


def detokenize(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""

    out: list[str] = []
    for tok in tokens:
        if not out:
            out.append(tok)
            continue

        # No space before punctuation like . , ; : ! ? )
        if re.match(r"^[\]\)\}.,;:!?%]$", tok):
            out[-1] = out[-1] + tok
        # No space after opening brackets/quotes
        elif out[-1] in {"(", "[", "{", "\"", "'", "¿", "¡"}:
            out[-1] = out[-1] + tok
        else:
            out.append(" " + tok)

    return normalize_whitespace("".join(out))


def count_word_tokens(tokens: Sequence[str]) -> int:
    return sum(1 for t in tokens if is_word_token(t))


def word_indices(tokens: Sequence[str]) -> list[int]:
    return [i for i, t in enumerate(tokens) if is_word_token(t)]


def n_modifications(num_words: int, intensity: float, *, min_n: int = 1) -> int:
    """Convert an intensity (0..1) to a number of word-level edits.

    Uses round(num_words * intensity), with a minimum of 1 when intensity>0.
    """
    if num_words <= 0:
        return 0
    intensity = float(max(0.0, min(1.0, intensity)))
    if intensity == 0.0:
        return 0
    n = int(round(num_words * intensity))
    return max(min_n, min(num_words, n))


def sample_indices(candidates: Sequence[int], k: int, *, rng: random.Random | None = None) -> list[int]:
    if k <= 0 or not candidates:
        return []
    k = min(k, len(candidates))
    rng = rng or random
    return list(rng.sample(list(candidates), k=k))


def preserve_case(src: str, dst: str) -> str:
    if not src:
        return dst
    if src[0].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def lower_strip(text: str) -> str:
    return normalize_whitespace(text).lower()


def extract_keywords(text: str) -> set[str]:
    """Cheap keyword set: lowercase word tokens excluding short tokens."""
    toks = [t.lower() for t in simple_tokenize(text) if is_word_token(t)]
    return {t for t in toks if len(t) >= 4}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))
