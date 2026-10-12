"""
FRED Evaluation — Step 3: LLM-as-a-Judge (FActScoreLite)
=========================================================
Uses Gemini 3.1 Flash as an LLM judge to evaluate factual accuracy
of responses, aligning with the FRED paper's Table 5 (Editing Results).

Two conditions are judged per sample:
  1. No-Edit:   the original corrupted response (baseline)
  2. FRED-Edit: the response after FRED's XML tag edits are applied

Budget: 500 sampled test records × 2 conditions = 1000 API calls
Keys:   GEMINI_API_KEY_1 (first 500 calls), GEMINI_API_KEY_2 (next 500)
Rate:   15 RPM per key (4.1s sleep between calls)

Output:
  - data/eval/judge_results.jsonl   (per-sample judgments)
  - data/eval/judge_scores.json     (aggregate scores)

Resume-safe: checkpoints every record, safe to ctrl+C and restart.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import os
import sys
import re
import time
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# ── Ensure UTF-8 on Windows ─────────────────────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Setup paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ── Config ───────────────────────────────────────────────────────────────────
EVAL_DIR       = ROOT_DIR / "data" / "eval"
TEST_PATH      = ROOT_DIR / "data" / "final" / "test.jsonl"
PRED_FRED_PATH = EVAL_DIR / "predictions_fred.jsonl"
JUDGE_OUT_PATH = EVAL_DIR / "judge_results.jsonl"
JUDGE_SCORES   = EVAL_DIR / "judge_scores.json"

SAMPLE_SIZE       = 500      # number of test records to sample
RATE_LIMIT_SLEEP  = 4.1      # seconds between calls (≈14.6 RPM, under 15 RPM)
CALLS_PER_KEY     = 500      # switch key after this many calls
GEMINI_MODEL      = "gemini-3.1-flash-lite"

RANDOM_SEED = 42  # reproducible sampling

# ── Judge Prompt ─────────────────────────────────────────────────────────────

JUDGE_PROMPT = """\
You are a financial fact-checking judge. Your task is to evaluate whether a \
RESPONSE to a financial question is factually accurate based ONLY on the \
provided CONTEXT (financial documents and tables).

EVALUATION CRITERIA:
1. Are all numerical values (dollar amounts, percentages, ratios) correct?
2. Are all temporal references (years, dates, time periods) correct?
3. Are all calculations and mathematical reasoning steps correct?
4. Does the final answer match what can be derived from the context?

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE TO EVALUATE:
{response}

SCORING INSTRUCTIONS:
Rate the factual accuracy of the RESPONSE on a scale of 1-5:
  1 = Completely wrong — major numerical/temporal errors or wrong answer
  2 = Mostly wrong — contains significant factual errors
  3 = Partially correct — some facts right but key errors present
  4 = Mostly correct — minor errors that don't affect the conclusion
  5 = Fully correct — all facts, numbers, and reasoning are accurate

You MUST respond with ONLY a JSON object in this exact format:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}

Do NOT include any other text, preamble, or markdown formatting."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_env_keys():
    """Load both Gemini API keys from .env file."""
    env_file = find_dotenv(usecwd=True)
    if not env_file:
        env_file = find_dotenv(filename=".env", raise_error_if_not_found=False)
    load_dotenv(env_file, override=True)

    key1 = os.environ.get("GEMINI_API_KEY_1", "")
    key2 = os.environ.get("GEMINI_API_KEY_2", "")

    if not key1:
        raise ValueError("GEMINI_API_KEY_1 not found in .env")
    if not key2:
        print("  WARNING: GEMINI_API_KEY_2 not found. Will use only KEY_1.")
        key2 = key1  # fallback to same key

    return key1, key2


def setup_gemini(api_key):
    """Configure the Gemini client with the given API key."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model


def apply_fred_edits(prediction_text):
    """
    Apply FRED's XML tag edits to produce the final cleaned response.
    Converts: <type><delete>wrong</delete><mark>right</mark></type> → right
    If no tags found, returns the text as-is (model predicted "no edit needed").
    """
    if not prediction_text:
        return prediction_text

    text = prediction_text
    # Strip "Correction: " prefix
    if text.lower().startswith("correction:"):
        text = text[len("correction:"):].strip()

    # Apply edits: replace <type><delete>X</delete><mark>Y</mark></type> with Y
    pattern = re.compile(
        r'<(numerical|temporal)>\s*<delete>.*?</delete>\s*<mark>(.*?)</mark>\s*</\1>',
        re.IGNORECASE | re.DOTALL
    )
    edited = pattern.sub(r'\2', text)

    # Clean any remaining stray tags
    edited = re.sub(r'</?(?:numerical|temporal|delete|mark)>', '', edited, flags=re.IGNORECASE)

    return edited.strip()


def build_context(test_record):
    """Build a readable context string from the test record's source documents."""
    docs_raw = test_record.get("source_documents", "")
    try:
        doc_list = json.loads(docs_raw) if isinstance(docs_raw, str) else docs_raw
    except (json.JSONDecodeError, TypeError):
        doc_list = [docs_raw] if docs_raw else []

    parts = []
    for i, d in enumerate(doc_list[:3]):
        d_str = str(d).strip()
        if len(d_str) > 800:
            d_str = d_str[:800] + "..."
        parts.append(f"[Doc {i+1}] {d_str}")
    return "\n\n".join(parts)


class DailyQuotaExceeded(Exception):
    pass


def call_judge(model, context, question, response_text, max_retries=3):
    """
    Call Gemini to judge a single response.
    Returns (score: int, reason: str) or (None, error_msg) on failure.
    Raises DailyQuotaExceeded if daily request limit is hit.
    """
    prompt = JUDGE_PROMPT.format(
        context=context,
        question=question,
        response=response_text,
    )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 200,
                }
            )
            raw = response.text.strip()

            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                result = json.loads(json_match.group())
                score = int(result.get("score", 0))
                reason = str(result.get("reason", ""))
                if 1 <= score <= 5:
                    return score, reason

            score_match = re.search(r'"score"\s*:\s*(\d)', raw)
            if score_match:
                score = int(score_match.group(1))
                if 1 <= score <= 5:
                    return score, raw

            print(f"      Parse failed (attempt {attempt+1}): {raw[:100]}", flush=True)

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            # Check for hard daily quota exceeded
            if ("per day" in err_lower or "resourceexhausted" in err_lower
                    or ("quota" in err_lower and "exceeded" in err_lower and "minute" not in err_lower)):
                print(f"      Daily quota exhausted: {err_str[:120]}", flush=True)
                raise DailyQuotaExceeded(err_str)

            # Transient rate limit (per minute RPM)
            if "429" in err_str or "quota" in err_lower or "rate" in err_lower:
                wait = 20 * (attempt + 1)
                print(f"      Rate limited (RPM), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if "500" in err_str or "503" in err_str:
                wait = 10 * (attempt + 1)
                print(f"      Server error, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"      API error (attempt {attempt+1}): {err_str[:100]}", flush=True)

        if attempt < max_retries - 1:
            time.sleep(5)

    return None, "FAILED_AFTER_RETRIES"


def load_checkpoint():
    """Load already-completed judge results for resume safety."""
    done = {}
    if JUDGE_OUT_PATH.exists():
        with open(JUDGE_OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    # Only treat as done if score is valid
                    if obj.get("score") is not None:
                        key = f"{obj['source_id']}_{obj['condition']}"
                        done[key] = obj
    return done


def load_test_data():
    """Load test records."""
    records = []
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_fred_predictions():
    """Load FRED predictions indexed by source_id."""
    preds = {}
    with open(PRED_FRED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                preds[obj["source_id"]] = obj
    return preds


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("FRED EVALUATION — LLM-AS-A-JUDGE (FActScoreLite)")
    print(f"Model: {GEMINI_MODEL}")
    print(f"Time:  {datetime.now().isoformat()}")
    print("=" * 72)

    # Load keys
    key1, key2 = load_env_keys()
    print(f"\n  API Keys loaded: KEY_1={'***'+key1[-4:]}, KEY_2={'***'+key2[-4:]}")

    # Load data
    test_data = load_test_data()
    fred_preds = load_fred_predictions()
    print(f"  Test records: {len(test_data)}")
    print(f"  FRED predictions: {len(fred_preds)}")

    # Sample records (deterministic)
    random.seed(RANDOM_SEED)
    sampled_ids = random.sample(
        [r["source_id"] for r in test_data],
        min(SAMPLE_SIZE, len(test_data))
    )
    sampled_records = {r["source_id"]: r for r in test_data if r["source_id"] in sampled_ids}
    print(f"  Sampled for judging: {len(sampled_ids)} records")

    # Count corrupted vs clean in sample
    n_corrupted = sum(1 for sid in sampled_ids if not sampled_records[sid].get("is_clean", False))
    n_clean = len(sampled_ids) - n_corrupted
    print(f"    Corrupted: {n_corrupted}, Clean: {n_clean}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"  Checkpoint: {len(checkpoint)} judgments already done")

    # Build task list: for each sampled record, 2 conditions
    tasks = []
    for sid in sampled_ids:
        rec = sampled_records[sid]
        context = build_context(rec)
        question = rec.get("question", "")

        # Condition 1: No-Edit (the corrupted response as-is)
        key_no_edit = f"{sid}_no_edit"
        if key_no_edit not in checkpoint:
            tasks.append({
                "source_id": sid,
                "condition": "no_edit",
                "context": context,
                "question": question,
                "response_to_judge": rec.get("response", ""),
                "is_clean": rec.get("is_clean", False),
                "error_type": rec.get("error_type", "none"),
            })

        # Condition 2: FRED-Edit (apply XML edits from FRED's prediction)
        key_fred = f"{sid}_fred_edit"
        if key_fred not in checkpoint:
            pred = fred_preds.get(sid, {})
            fred_prediction = pred.get("prediction", "")
            # Apply the XML edits to get the final text
            fred_edited_response = apply_fred_edits(fred_prediction)
            tasks.append({
                "source_id": sid,
                "condition": "fred_edit",
                "context": context,
                "question": question,
                "response_to_judge": fred_edited_response,
                "is_clean": rec.get("is_clean", False),
                "error_type": rec.get("error_type", "none"),
            })

    total_tasks = len(tasks)
    print(f"\n  Remaining tasks: {total_tasks}")

    if total_tasks == 0:
        print("  All judgments complete! Run scripts/15_score_llm_judge.py to generate report.")
        return

    # Setup Gemini with active key
    key1_done = sum(1 for v in checkpoint.values() if v.get("api_key_used") == 1)
    key2_done = sum(1 for v in checkpoint.values() if v.get("api_key_used") == 2)
    print(f"  Existing calls: Key 1 = {key1_done}, Key 2 = {key2_done}", flush=True)

    if key1_done >= 490:
        print(f"  Key 1 quota used ({key1_done} calls). Starting directly with Key 2.", flush=True)
        current_key_idx = 2
        gemini_model = setup_gemini(key2)
        call_count = key2_done
    else:
        current_key_idx = 1
        gemini_model = setup_gemini(key1)
        call_count = key1_done

    print(f"  Active API Key: Key {current_key_idx}", flush=True)
    print(f"\n{'-' * 72}")
    print("  RUNNING JUDGMENTS")
    print(f"{'-' * 72}\n", flush=True)

    # Open output in append mode
    with open(JUDGE_OUT_PATH, "a", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            # Check if Key 1 call count crossed threshold
            if current_key_idx == 1 and call_count >= 495:
                print(f"\n  >>> Switching to API Key 2 (reached {call_count} calls on Key 1)", flush=True)
                gemini_model = setup_gemini(key2)
                current_key_idx = 2
                call_count = 0
                time.sleep(2)

            # Call the judge with automatic key rotation on quota exhaustion
            t0 = time.time()
            score = None
            reason = ""
            for attempt_key in range(2):
                try:
                    score, reason = call_judge(
                        gemini_model,
                        task["context"],
                        task["question"],
                        task["response_to_judge"],
                    )
                    break
                except DailyQuotaExceeded:
                    if current_key_idx == 1:
                        print(f"\n  >>> Daily quota reached on Key 1. Immediately rotating to Key 2!", flush=True)
                        gemini_model = setup_gemini(key2)
                        current_key_idx = 2
                        call_count = 0
                        time.sleep(2)
                        continue
                    elif current_key_idx == 2 and attempt_key == 0:
                        print(f"\n  >>> Daily quota reached on Key 2. Trying Key 1 (quota reset overnight)!", flush=True)
                        gemini_model = setup_gemini(key1)
                        current_key_idx = 1
                        call_count = 0
                        time.sleep(2)
                        continue
                    else:
                        print(f"\n  >>> Daily quota reached on all available keys.", flush=True)
                        score, reason = None, "QUOTA_EXCEEDED_ALL_KEYS"
                        break

            elapsed = time.time() - t0
            call_count += 1

            # Only persist if we got a valid score or all keys exhausted
            result = {
                "source_id": task["source_id"],
                "condition": task["condition"],
                "is_clean": task["is_clean"],
                "error_type": task["error_type"],
                "score": score,
                "reason": reason,
                "elapsed_s": round(elapsed, 2),
                "api_key_used": current_key_idx,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            # Progress
            done_total = i + 1
            pct = done_total / total_tasks * 100
            score_str = f"score={score}" if score else "FAILED"
            remaining = total_tasks - done_total
            eta_min = remaining * RATE_LIMIT_SLEEP / 60

            if done_total % 5 == 0 or done_total == total_tasks or score is None:
                print(f"  [{done_total:4d}/{total_tasks}] ({pct:5.1f}%) "
                      f"{task['source_id']:15s} {task['condition']:10s} "
                      f"{score_str:8s} | Key{current_key_idx} "
                      f"| {elapsed:.1f}s | ~{eta_min:.0f}m left", flush=True)

            # Rate limiting
            if done_total < total_tasks:
                sleep_time = max(0, RATE_LIMIT_SLEEP - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("JUDGING COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Results saved to: {JUDGE_OUT_PATH}")
    print(f"  Total judgments: {total_tasks + len(checkpoint)}")
    print(f"\n  Next step: python scripts/15_score_llm_judge.py")


if __name__ == "__main__":
    main()
