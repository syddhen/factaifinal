"""Small, dependency-free presentation helpers for the FactAI interface."""

VERDICT_TONES = {
    "True": {"label": "Supported", "accent": "#0F766E", "icon": "✓"},
    "False": {"label": "Contradicted", "accent": "#B42318", "icon": "×"},
    "Misleading": {"label": "Mixed context", "accent": "#B54708", "icon": "!"},
    "Unverified": {"label": "Needs review", "accent": "#475467", "icon": "?"},
}

HISTORY_PROMPTS = {
    "Science": "The Earth revolves around the Sun.",
    "Health": "Antibiotics do not work against viruses.",
    "Technology": "The first iPhone was released in 2007.",
    "Climate": "Global average sea level has risen since 1900.",
}

MIN_CLAIM_LENGTH = 5


def verdict_tone(verdict: str) -> dict[str, str]:
    """Return the presentational tone for a pipeline verdict."""
    return VERDICT_TONES.get(verdict, VERDICT_TONES["Unverified"])


def history_prompt(topic: str) -> str:
    """Return the text restored when a front-end history item is selected."""
    return HISTORY_PROMPTS.get(topic, HISTORY_PROMPTS["Science"])


def valid_claim_input(text: str) -> bool:
    """The UI and HTTP API accept the same minimum claim length."""
    return len((text or "").strip()) >= MIN_CLAIM_LENGTH
