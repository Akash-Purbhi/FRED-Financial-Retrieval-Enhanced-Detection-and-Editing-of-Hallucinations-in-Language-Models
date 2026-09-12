"""
config.py — Central config for all FRED notebooks.
Change anything here; notebooks import from this file.
"""

# ── Groq model priority ────────────────────────────────────────────────────
# First model in this list that is available on your account will be used.
PREFERRED_MODELS = [
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

# ── Error types ────────────────────────────────────────────────────────────
ERROR_TYPES = [
    "Temporal",
    "Numerical",
    "Entity",
    "Relation",
    "Contradictory",
    "Unverifiable",
]

# ── Dataset ────────────────────────────────────────────────────────────────
RAGBENCH_REPO  = "rungalileo/ragbench"
FINQA_SUBSET   = "finqa"
TATQA_SUBSET   = "tatqa"

# ── Generation ────────────────────────────────────────────────────────────
TEMPERATURE = 1.0
MAX_TOKENS  = 1024
