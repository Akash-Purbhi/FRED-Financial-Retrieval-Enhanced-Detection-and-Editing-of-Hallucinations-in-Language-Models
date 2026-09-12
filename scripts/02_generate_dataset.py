"""
02_generate_dataset.py
Generate a dataset of exactly 1,500 valid synthetic hallucinations on FinQA.
Runs locally using Ollama with Qwen2.5:7b and FRED quality filtering.
Automatically resumes from existing rows in synthetic_finqa_1500.jsonl.
"""

import os
import sys
import json
import time
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset

# Ensure workspace root is in sys.path when running from scripts/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import groq_utils as gu
from filter_data import filter_generated_data
from config import RAGBENCH_REPO, FINQA_SUBSET

MODEL = "qwen2.5:7b"
TARGET_COUNT = 1500
OUTPUT_FILE = ROOT_DIR / "synthetic_finqa_1500.jsonl"


def main():
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    print(f"Loading dataset {RAGBENCH_REPO} [{FINQA_SUBSET}]...")
    dataset = load_dataset(RAGBENCH_REPO, FINQA_SUBSET, trust_remote_code=True)
    train_data = dataset["train"]

    # Load existing samples to resume progress without losing previous work
    valid_samples = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        valid_samples.append(json.loads(line))
                    except Exception:
                        pass
        if valid_samples:
            print(f"Resuming: found {len(valid_samples)} existing valid rows.")

    print(f"Target: {TARGET_COUNT} valid samples using {MODEL}...\n")

    start_time = time.time()
    initial_count = len(valid_samples)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for idx, row in enumerate(train_data):
            if len(valid_samples) >= TARGET_COUNT:
                break

            # Skip rows already processed in previous runs
            if idx < initial_count:
                continue

            documents = row.get("documents") or row.get("document") or row.get("context") or ""
            question = row.get("question") or row.get("query") or ""
            original_response = row.get("response") or row.get("answer") or row.get("output") or ""

            if not original_response.strip():
                continue

            tagged_response = gu.insert_error(
                client, MODEL, documents, question, original_response
            )

            is_valid, reason = filter_generated_data(original_response, tagged_response)

            if not is_valid:
                continue

            record = {
                "documents": documents,
                "question": question,
                "original_response": original_response,
                "tagged_response": tagged_response,
            }

            valid_samples.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if len(valid_samples) % 10 == 0:
                elapsed = time.time() - start_time
                new_collected = len(valid_samples) - initial_count
                avg_time = (elapsed / new_collected) if new_collected > 0 else 0
                print(
                    f"Progress: {len(valid_samples)} / {TARGET_COUNT} valid rows collected "
                    f"(evaluated {idx + 1} rows) | Avg: {avg_time:.2f}s/valid row"
                )

    total_time = time.time() - start_time
    print(f"\nDataset generation complete in {total_time:.2f}s.")
    print(f"Total {len(valid_samples)} valid rows securely saved in {OUTPUT_FILE.name}.\n")


if __name__ == "__main__":
    main()
