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

# Generation settings
TEMPERATURE = 1.0
MAX_TOKENS = 900
TARGET_SAMPLES = 1500
OUTPUT_FILE = "synthetic_finqa_generated.jsonl"
