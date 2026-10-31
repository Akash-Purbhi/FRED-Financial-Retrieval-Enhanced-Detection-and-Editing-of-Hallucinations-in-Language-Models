"""
scripts/17_run_local_ollama.py
==============================
Command-line test runner for local offline inference with Ollama.

Usage:
  python scripts/17_run_local_ollama.py --model fred-stage2 --samples 20
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
import requests

WORKSPACE = Path(__file__).resolve().parent.parent
OLLAMA_BASE_URL = "http://localhost:11434"


def query_ollama(model_name: str, context: str, table: list, question: str, candidate: str):
    table_str = "\n".join([" | ".join(r) for r in table]) if table else "[No Table]"
    user_content = (
        f"Context: {context}\n\n"
        f"Table:\n{table_str}\n\n"
        f"Question: {question}\n\n"
        f"Response to Verify: {candidate}"
    )

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_content}],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9},
    }

    start = time.time()
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
    dur = time.time() - start

    if resp.status_code == 200:
        data = resp.json()
        out = data["message"]["content"].strip()
        toks = data.get("eval_count", 0)
        eval_ns = data.get("eval_duration", 1)
        tps = (toks / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
        return out, dur, tps
    else:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="FRED Stage 2 Local Ollama Evaluator")
    parser.add_argument("--model", type=str, default="fred-stage2", help="Ollama model name")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples")
    args = parser.parse_args()

    print(f"Connecting to local Ollama at {OLLAMA_BASE_URL} (Model: {args.model})...")
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            print("Ollama server not healthy!")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot reach Ollama server: {e}")
        print("Please start Ollama using: ollama serve")
        sys.exit(1)

    test_clean_file = WORKSPACE / "data" / "stage2" / "test_clean.jsonl"
    with open(test_clean_file, "r", encoding="utf-8") as f:
        clean_records = [json.loads(line) for line in f][:args.samples]

    print(f"Running clean evaluation on {len(clean_records)} samples...")
    preserved = 0
    tps_list = []

    for idx, sample in enumerate(clean_records, 1):
        out, lat, tps = query_ollama(
            args.model,
            sample.get("context", ""),
            sample.get("table", []),
            sample.get("question", ""),
            sample.get("unverified_response", sample.get("gold_answer", "")),
        )
        tps_list.append(tps)
        is_preserved = '"action": "preserve"' in out or "preserve" in out[:30]
        if is_preserved:
            preserved += 1

        print(f"[{idx}/{len(clean_records)}] Latency: {lat:.2f}s ({tps:.1f} tok/s) | Preserved: {is_preserved}")

    cpr = (preserved / len(clean_records)) * 100.0
    avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0.0
    print(f"\nResults for {args.model}:")
    print(f"  Clean Preservation Rate (CPR): {cpr:.1f}%")
    print(f"  Average Throughput:             {avg_tps:.1f} tokens/second")


if __name__ == "__main__":
    main()
