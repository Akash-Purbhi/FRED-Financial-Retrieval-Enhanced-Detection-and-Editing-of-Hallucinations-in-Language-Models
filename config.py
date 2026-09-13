"""
config.py
Central configuration for datasets, models, and generation parameters.
"""

# Datasets
RAGBENCH_REPO = "rungalileo/ragbench"
FINQA_SUBSET = "finqa"
TATQA_SUBSET = "tatqa"

# Generation models
GEMINI_MODEL = "gemini-3.1-flash-lite"
PREFERRED_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]

# Backward compatibility alias for groq_utils
PREFERRED_MODELS = PREFERRED_GROQ_MODELS

# Valid FRED error types
ERROR_TYPES = [
    "temporal",
    "numerical",
    "entity",
    "relation",
    "contradictory",
    "unverifiable",
]

# Generation settings
TEMPERATURE = 1.0
MAX_TOKENS = 900
TARGET_SAMPLES = 1500
OUTPUT_FILE = "synthetic_finqa_1500.jsonl"
