"""
03_generate_dataset_gemini.py
Generate synthetic hallucination dataset on FinQA using Google Gemini API (gemini-2.5-flash-lite).
Enforces FRED quality filtering, 15 RPM rate limiting, and 990 daily call circuit breaker.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from datasets import load_dataset

# Ensure workspace root is in sys.path when running from scripts/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from filter_data import filter_generated_data
from config import RAGBENCH_REPO, FINQA_SUBSET

MODEL_NAME = "gemini-3.5-flash-lite"
TARGET_COUNT = 1500
OUTPUT_FILE = ROOT_DIR / "synthetic_finqa_1500.jsonl"
RATE_LIMIT_SLEEP_SEC = 4.5   # Strictly comply with 15 RPM limit
DAILY_CALL_LIMIT = 990       # Daily quota circuit breaker


PROMPT_TEMPLATE = """\
You are a dataset-annotation assistant. Your task is to insert exactly ONE factual error \
into the RESPONSE below. Use the reference DOCUMENTS and QUESTION only to understand context \
— do NOT alter them.

Choose ONE error type at random from this list:
  temporal, numerical, entity, relation, contradictory, unverifiable

Tagging rules (follow exactly, no exceptions):
  • Span-level errors (temporal / numerical / entity / relation):
      <type><delete>original_span</delete><mark>corrupted_span</mark></type>
    Replace type with the chosen error type.
  • Sentence-level errors (contradictory / unverifiable):
      <type>corrupted_sentence</type>
    The corrupted sentence must replace the original sentence entirely inside the tag.

  Use strictly lowercase for the tag names (e.g., <numerical>, <temporal>). Do not use capital letters.

Constraints:
  - Insert EXACTLY one error. No more.
  - Output ONLY the tagged corrupted response. No explanation, no preamble.
  - Do NOT change any part of the response that is not involved in the error.
  - Do NOT add new sentences. Only corrupt or replace an existing span or sentence.

---
DOCUMENTS:
{documents}

QUESTION:
{question}

RESPONSE:
{response}
---

CRITICAL CONSTRAINT: Do NOT output any chain-of-thought, reasoning, or <think> blocks. Your entire output must consist ONLY of the final tagged corrupted response.

Output the tagged corrupted response now:"""


def load_gemini_api_key():
    """Load GEMINI_API_KEY from .env (or GOOGLE_API_KEY fallback)."""
    env_file = find_dotenv(usecwd=True)
    load_dotenv(env_file, override=True)
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY not found in .env file. Please add GEMINI_API_KEY='your_key' to FRED/.env")
    return key


def format_documents(docs):
    if isinstance(docs, list):
        return "\n\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(docs))
    return str(docs)


def insert_error(model, documents_raw, question, response):
    """
    Call Gemini generate_content to insert one tagged error.
    Catches timeouts, safety blocks, or exceptions and returns empty string.
    """
    prompt = PROMPT_TEMPLATE.format(
        documents=format_documents(documents_raw),
        question=question,
        response=response,
    )
    try:
        completion = model.generate_content(prompt)
        text = completion.text.strip() if completion and completion.text else ""
        text = re.sub(r'<think>.*?</think>\n*', '', text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        print(f"API generation error: {e}")
        return ""


def main():
    api_key = load_gemini_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    print(f"Loading dataset {RAGBENCH_REPO} [{FINQA_SUBSET}]...")
    dataset = load_dataset(RAGBENCH_REPO, FINQA_SUBSET, trust_remote_code=True)
    train_data = dataset["train"]

    # Load existing samples to resume progress
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

    print(f"Target: {TARGET_COUNT} valid samples using Gemini ({MODEL_NAME})...\n")

    start_time = time.time()
    initial_count = len(valid_samples)
    api_call_count = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for idx, row in enumerate(train_data):
            if len(valid_samples) >= TARGET_COUNT:
                break

            # Check daily quota circuit breaker
            if api_call_count >= DAILY_CALL_LIMIT:
                print("\nDaily limit reached. Pausing until tomorrow.\n")
                break

            # Skip rows already processed in previous runs
            if idx < initial_count:
                continue

            documents = row.get("documents") or row.get("document") or row.get("context") or ""
            question = row.get("question") or row.get("query") or ""
            original_response = row.get("response") or row.get("answer") or row.get("output") or ""

            if not original_response.strip():
                continue

            api_call_count += 1
            tagged_response = insert_error(model, documents, question, original_response)

            # Rate-limiting sleep to strictly adhere to 15 RPM
            time.sleep(RATE_LIMIT_SLEEP_SEC)

            if not tagged_response:
                continue

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
                    f"(calls: {api_call_count}) | Avg: {avg_time:.2f}s/valid row"
                )

    total_time = time.time() - start_time
    print(f"\nRun complete in {total_time:.2f}s.")
    print(f"Total API calls made this run: {api_call_count}")
    print(f"Total {len(valid_samples)} valid rows saved in {OUTPUT_FILE.name}.\n")


if __name__ == "__main__":
    main()
