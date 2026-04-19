from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PERTURBED_DATA_DIR = DATA_DIR / "perturbed"

DEFAULT_DATASET = RAW_DATA_DIR / "sample_dataset.json"

# --- Attack settings ---
AVAILABLE_ATTACKS = ["synonym", "paraphrase", "minimal_pair", "shortcut_removal"]

INTENSITY_LEVELS = {
    "low": 0.1,
    "medium": 0.3,
    "high": 0.5,
}

DEFAULT_INTENSITY = INTENSITY_LEVELS["medium"]

# --- Model settings ---
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

DEFAULT_MODEL = "gpt-4o-mini"

# --- Evaluation settings ---
SEMANTIC_SIMILARITY_THRESHOLD = 0.85
