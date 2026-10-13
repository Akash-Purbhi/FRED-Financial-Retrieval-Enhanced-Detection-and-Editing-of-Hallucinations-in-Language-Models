"""
FRED Evaluation — Step 1: Run Inference
========================================
Sends all 841 test records to both models via Ollama API.
Saves raw predictions with checkpointing (resume-safe).

Models:
  - fred:latest       (fine-tuned)
  - qwen2.5:7b        (base)

Output:
  - data/eval/predictions_fred.jsonl
  - data/eval/predictions_base.jsonl
"""

import json
import os
import sys
import time
import shutil
import argparse
import requests
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
TEST_PATH  = Path("data/final/test.jsonl")
EVAL_DIR   = Path("data/eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "fred":  "fred:latest",
    "base":  "qwen2.5:7b",
}

CHECKPOINT_INTERVAL = 10  # save every N records

BASE_FEW_SHOT_TEMPLATE = """Given a passage with factual errors, identify any <numerical> or <temporal> errors in the passage and add edits for <numerical> and <temporal> errors by inserting additional <mark></mark> or <delete></delete> tags to mark and delete. If there are no errors, return the passage with no tags. Any changes to the original passage should be marked in <> tags. Below are the error definitions followed by an example of what you need to follow.

Definitions:
1. numerical errors (<numerical>): an incorrect calculation, estimation, or interpretation of numerical data such as percentages, growth rates, totals, differences, or ratios. These errors can arise from misapplying formulas, misreading data, rounding incorrectly, or failing to consider time periods or units.
2. temporal errors (<temporal>): incorrect reference or use of figures from the wrong time period. These errors typically arise from misinterpreting the reference's temporal context, such as year-over-year comparisons or quarter-specific data.

Follow the given example exactly, your task is to create the edited completion with error tags <>:

Passage: Acme Corp's revenue reached $1.4 billion in Q4 2023, a 18% increase compared to the same quarter in 2021. The company posted a net income of $450 million, up from $390 million the year before. Acme attributed the improved financial performance to strong demand for its cloud services division and successful restructuring efforts carried out earlier in the year.

Reference: In Q4 2023, Acme Corp reported a revenue of $2.4 billion, marking a 12% increase from Q4 2022. Net income for the quarter stood at $450 million, up from $390 million in the previous year. The company attributed this growth to increased demand in its cloud services division and operational efficiencies gained through restructuring efforts implemented in early 2023.

Edited: Acme Corp's revenue reached <numerical><delete>$1.4</delete><mark>$2.4</mark></numerical> billion in Q4 2023, a <numerical><delete>18%</delete><mark>12%</mark></numerical> increase compared to the same quarter in <temporal><delete>2021</delete><mark>2022</mark></temporal>. The company posted a net income of $450 million, up from $390 million the year before. Acme attributed the improved financial performance to strong demand for its cloud services division and successful restructuring efforts carried out earlier in the year.

Now detect errors and include edits in the following passage like done in the example above. Include error tags <> for ANYTHING YOU CHANGE IN THE ORIGINAL PASSAGE. If there are no errors, return the original passage unchanged without any tags.

Passage: {passage}
Reference: {reference}

Return valid JSON in the following format:
{{"Edited": "paragraph with inserted errors"}}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_test_data(limit=None):
    records = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def extract_context(record):
    docs = record.get("source_documents", "")
    try:
        doc_list = json.loads(docs) if isinstance(docs, str) else docs
    except (json.JSONDecodeError, TypeError):
        doc_list = [docs] if docs else []

    context_parts = []
    for d in doc_list[:3]:
        d_str = str(d).strip()
        if len(d_str) > 600:
            d_str = d_str[:600] + "..."
        context_parts.append(d_str)
    return "\n\n".join(context_parts)


def build_prompt_fred(record):
    """Build the ChatML prompt format used during training."""
    context = extract_context(record)
    question = record.get("question", "")
    response = record.get("response", "")
    return f"Context: {context}\nQuestion: {question}\nResponse: {response}"


def build_prompt_base(record):
    """Build the paper Appendix E.2 few-shot prompt tailored to numerical and temporal errors."""
    context = extract_context(record)
    question = record.get("question", "")
    reference = f"Question: {question}\nContext: {context}"
    passage = record.get("response", "")
    return BASE_FEW_SHOT_TEMPLATE.format(passage=passage, reference=reference)


def build_prompt(record, model_key):
    if model_key == "base":
        return build_prompt_base(record)
    return build_prompt_fred(record)


def query_ollama(model_name, prompt, timeout=120):
    """Send a prompt to Ollama and return the generated text."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1024,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def load_checkpoint(path):
    """Load already-completed predictions from checkpoint file."""
    done = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    done[obj["source_id"]] = obj
    return done


def run_inference(model_key, model_name, test_data, fresh=False):
    """Run inference for one model across all test records."""
    out_path = EVAL_DIR / f"predictions_{model_key}.jsonl"
    
    if fresh and out_path.exists():
        backup_path = EVAL_DIR / f"predictions_{model_key}_backup.jsonl"
        print(f"  [{model_key}] --fresh requested. Backing up existing file to {backup_path.name}")
        shutil.copy2(out_path, backup_path)
        out_path.unlink()

    checkpoint = load_checkpoint(out_path)
    remaining = [r for r in test_data if r["source_id"] not in checkpoint]
    total = len(test_data)
    done_count = len(checkpoint)

    if not remaining:
        print(f"  [{model_key}] All {total} records already completed. Skipping.")
        return

    print(f"  [{model_key}] {done_count}/{total} done. Running {len(remaining)} remaining...")

    # Open in append mode
    with open(out_path, "a", encoding="utf-8") as f:
        for i, record in enumerate(remaining):
            prompt = build_prompt(record, model_key)
            t0 = time.time()
            prediction = query_ollama(model_name, prompt)
            elapsed = time.time() - t0

            result = {
                "source_id": record["source_id"],
                "error_type": record["error_type"],
                "is_clean": record["is_clean"],
                "target_output": record["target_output"],
                "response": record["response"],
                "source_response": record["source_response"],
                "prediction": prediction,
                "elapsed_s": round(elapsed, 2),
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            current = done_count + i + 1
            if current % CHECKPOINT_INTERVAL == 0 or current == total:
                f.flush()
                avg_time = elapsed
                eta_s = avg_time * (total - current)
                eta_min = eta_s / 60
                print(f"    [{model_key}] {current}/{total}  "
                      f"({elapsed:.1f}s this record, ~{eta_min:.0f}m remaining)")

    print(f"  [{model_key}] Complete. {total} predictions saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FRED Evaluation — Step 1: Run Inference")
    parser.add_argument(
        "--model",
        choices=["all", "fred", "base"],
        default="all",
        help="Which model to run inference for (default: all)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh (backs up and overwrites existing predictions for the specified model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test records to process (for quick validation)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("FRED EVALUATION — INFERENCE")
    print(f"Target: model={args.model}, fresh={args.fresh}, limit={args.limit}")
    print("=" * 72)

    # Check Ollama is running
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        print(f"\nOllama models available: {available}")
    except Exception as e:
        print(f"\nERROR: Cannot connect to Ollama at localhost:11434")
        print(f"       Make sure Ollama is running. Error: {e}")
        sys.exit(1)

    models_to_run = {}
    if args.model in ("all", "fred"):
        models_to_run["fred"] = MODELS["fred"]
    if args.model in ("all", "base"):
        models_to_run["base"] = MODELS["base"]

    # Verify target models exist
    for key, name in models_to_run.items():
        if not any(name in m for m in available):
            print(f"  WARNING: Model '{name}' not found in Ollama. Available: {available}")

    # Load test data
    test_data = load_test_data(limit=args.limit)
    print(f"\nTest records to evaluate: {len(test_data)}")
    print(f"  Corrupted: {sum(1 for r in test_data if not r['is_clean'])}")
    print(f"  Clean:     {sum(1 for r in test_data if r['is_clean'])}")

    # Run inference
    for model_key, model_name in models_to_run.items():
        print(f"\n{'-' * 40}")
        print(f"Running inference: {model_name} (key: {model_key})")
        print(f"{'-' * 40}")
        run_inference(model_key, model_name, test_data, fresh=args.fresh)

    print(f"\n{'=' * 72}")
    print("INFERENCE COMPLETE")
    print(f"{'=' * 72}")
    print(f"\nPredictions saved to:")
    for key in models_to_run:
        print(f"  data/eval/predictions_{key}.jsonl")


if __name__ == "__main__":
    main()

