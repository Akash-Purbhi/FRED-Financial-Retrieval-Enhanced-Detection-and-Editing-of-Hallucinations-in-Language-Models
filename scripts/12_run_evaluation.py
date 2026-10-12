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


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_test_data():
    records = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_prompt(record):
    """Build the same prompt format used during training."""
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
    context = "\n\n".join(context_parts)

    question = record.get("question", "")
    response = record.get("response", "")

    return f"Context: {context}\nQuestion: {question}\nResponse: {response}"


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


def run_inference(model_key, model_name, test_data):
    """Run inference for one model across all test records."""
    out_path = EVAL_DIR / f"predictions_{model_key}.jsonl"
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
            prompt = build_prompt(record)
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
                avg_time = elapsed  # just the last one for display
                eta_s = avg_time * (total - current)
                eta_min = eta_s / 60
                print(f"    [{model_key}] {current}/{total}  "
                      f"({elapsed:.1f}s this record, ~{eta_min:.0f}m remaining)")

    print(f"  [{model_key}] Complete. {total} predictions saved to {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("FRED EVALUATION — INFERENCE")
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

    # Verify both models exist
    for key, name in MODELS.items():
        if not any(name in m for m in available):
            print(f"  WARNING: Model '{name}' not found in Ollama. "
                  f"Available: {available}")

    # Load test data
    test_data = load_test_data()
    print(f"\nTest records: {len(test_data)}")
    print(f"  Corrupted: {sum(1 for r in test_data if not r['is_clean'])}")
    print(f"  Clean:     {sum(1 for r in test_data if r['is_clean'])}")

    # Run inference for each model
    for model_key, model_name in MODELS.items():
        print(f"\n{'-' * 40}")
        print(f"Running inference: {model_name}")
        print(f"{'-' * 40}")
        run_inference(model_key, model_name, test_data)

    print(f"\n{'=' * 72}")
    print("INFERENCE COMPLETE")
    print(f"{'=' * 72}")
    print(f"\nPredictions saved to:")
    for key in MODELS:
        print(f"  data/eval/predictions_{key}.jsonl")


if __name__ == "__main__":
    main()
