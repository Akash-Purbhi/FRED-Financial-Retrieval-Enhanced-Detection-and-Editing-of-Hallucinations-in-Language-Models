"""
05_generate_cot_dataset.py
Production-grade bulk generator for FRED synthetic financial CoT dataset (Target: 5,000 samples).

Key Architectural Rules:
1. Trimmed Document Formatting: Max 1,200 chars total context to guarantee 2-3s response time and prevent Google inference hangs.
2. Row-Level Fault Tolerance: If an outlier row times out twice, skip that row (row_idx += 1) without abandoning the model.
3. Strict Sequential Key & Model Progression:
   - Key 2 runs first (Model 1 -> Model 2 -> ...).
   - Key 1 runs next (Model 1 -> Model 2 -> ...).
4. Rate Limiting: 4.2s delay between successful calls (14 RPM, strictly respecting Google's 15 RPM free-tier limit).
5. 30s Cooldown on 429: Distinguishes temporary RPM rate limits from hard daily quota.
"""

import os
import sys
import json
import time
import re
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUTPUT_FILE = ROOT_DIR / "synthetic_finqa_cot_5000.jsonl"
TARGET_SAMPLES = 5000
MAX_SESSION_REQUESTS = 500
RATE_LIMIT_SLEEP_SEC = 4.5  # Strictly 15 RPM limit (time.sleep(4.5))
CALL_TIMEOUT_SEC = 45

MODELS = [
    "gemini-3.1-flash-lite",    # User requested gemini-3.1-flash-lite on Key 3
]

PROMPT_TEMPLATE = """\
You are an expert financial dataset annotation system.
Given the financial context documents and a financial question with its correct factual answer, you must generate a synthetic training sample for hallucination detection and correction.

FINANCIAL CONTEXT:
{context}

QUESTION:
{question}

GROUND TRUTH FACTUAL ANSWER:
{ground_truth}

TASK:
1. Create an erroneous RESPONSE that a model might generate. The response must contain exactly ONE deliberate NUMERICAL error (e.g., miscalculation, wrong dollar amount, incorrect percentage, swapped figure) OR ONE deliberate TEMPORAL error (e.g., wrong year, wrong quarter, wrong month).
2. Create a TARGET_OUTPUT that:
   - Starts with step-by-step mathematical calculations explaining the correct derivation from the context and identifying the specific error in the response.
   - Ends with "Correction: " followed by the full corrected response, wrapping the error span in:
     <numerical><delete>wrong_span</delete><mark>correct_span</mark></numerical>
     OR
     <temporal><delete>wrong_span</delete><mark>correct_span</mark></temporal>.

OUTPUT FORMAT:
Output strictly a single valid JSON object (no markdown formatting, no code blocks, no other text) with these exact keys:
{{
  "context": "Relevant financial context excerpt or table summary",
  "question": "{question_escaped}",
  "response": "The response containing the wrong numerical or temporal span",
  "target_output": "Reasoning: [step-by-step mathematical calculations]. Correction: [full sentence with <numerical><delete>wrong</delete><mark>correct</mark></numerical> or <temporal><delete>wrong</delete><mark>correct</mark></temporal>]"
}}
"""

def format_docs(docs):
    if isinstance(docs, list):
        formatted = []
        for i, d in enumerate(docs[:2]):
            d_str = str(d).strip()
            if len(d_str) > 550:
                d_str = d_str[:550] + "..."
            formatted.append(f"[Doc {i+1}] {d_str}")
        return "\n\n".join(formatted)[:1200]
    return str(docs).strip()[:1200]

def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def validate_sample(sample: dict) -> bool:
    required_keys = ["context", "question", "response", "target_output"]
    if not all(k in sample and sample[k] for k in required_keys):
        return False
    
    target = sample["target_output"]
    resp = sample["response"]
    
    if not ("Reasoning:" in target or "calculate" in target.lower() or "=" in target or "+" in target or "-" in target or "%" in target):
        return False
    
    tag_match = re.search(r"<(numerical|temporal)><delete>(.*?)</delete><mark>(.*?)</mark></\1>", target, re.IGNORECASE | re.DOTALL)
    if not tag_match:
        return False
    
    wrong = tag_match.group(2).strip()
    correct = tag_match.group(3).strip()
    
    if not wrong or not correct or wrong.lower() == correct.lower():
        return False
    
    if wrong.lower() not in resp.lower():
        return False
        
    return True

def call_model_with_timeout(model, prompt, timeout_sec=CALL_TIMEOUT_SEC):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(model.generate_content, prompt)
    try:
        return future.result(timeout=timeout_sec)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

def main():
    load_dotenv(find_dotenv(usecwd=True), override=True)
    
    # User requested: API Key 3, model gemini-3.1-flash-lite, 500 requests/day, 4.5s delay
    api_key = os.environ.get("GEMINI_API_KEY_3", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY_3 not found in environment (.env). Please add GEMINI_API_KEY_3=\"...\" to .env")
    
    # Count existing rows
    existing_count = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_count += 1
        print(f"Resuming: found {existing_count} existing rows in {OUTPUT_FILE.name}", flush=True)
    
    if existing_count >= TARGET_SAMPLES:
        print(f"Target of {TARGET_SAMPLES} samples already reached! Exiting.", flush=True)
        return

    print("Loading FinQA dataset from RagBench...", flush=True)
    dataset = load_dataset("rungalileo/ragbench", "finqa")
    train_data = dataset["train"]
    
    model_name = MODELS[0]
    row_idx = existing_count
    total_samples = existing_count
    row_retry_count = 0
    session_requests = 0
    model_429_attempts = 0
    
    print(f"Starting generation up to {TARGET_SAMPLES} samples (capped at {MAX_SESSION_REQUESTS} requests today)...")
    print(f"Model: {model_name} on Key 3")
    print(f"Rate Limiting: {RATE_LIMIT_SLEEP_SEC}s pacing (strictly within 15 RPM limit), Timeout: {CALL_TIMEOUT_SEC}s", flush=True)
    
    genai.configure(api_key=api_key, transport="rest")
    current_model = genai.GenerativeModel(model_name)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        while total_samples < TARGET_SAMPLES and row_idx < len(train_data):
            if session_requests >= MAX_SESSION_REQUESTS:
                print(f"\n[INFO] Reached requested daily cap of {MAX_SESSION_REQUESTS} requests for Key 1. Stopping generator.", flush=True)
                break
                
            row = train_data[row_idx]
            context_raw = row.get("documents") or row.get("context") or ""
            question = row.get("question") or ""
            answer = row.get("response") or row.get("answer") or ""
            
            if not context_raw or not question or not answer:
                row_idx += 1
                continue
                
            context_str = format_docs(context_raw)
            prompt = PROMPT_TEMPLATE.format(
                context=context_str,
                question=question,
                ground_truth=answer,
                question_escaped=question.replace('"', '\\"')
            )
            
            try:
                session_requests += 1
                response = call_model_with_timeout(current_model, prompt, timeout_sec=CALL_TIMEOUT_SEC)
                raw_text = clean_json_text(response.text)
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1:
                    raw_text = raw_text[start:end+1]
                data = json.loads(raw_text, strict=False)
                
                if not data.get("context") or len(data["context"]) < 30:
                    data["context"] = context_str[:600]
                data["question"] = question
                
                if validate_sample(data):
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    outfile.flush()
                    total_samples += 1
                    row_idx += 1
                    row_retry_count = 0
                    model_429_attempts = 0
                    print(f"[{total_samples}/{TARGET_SAMPLES}] ({model_name} [Key 1] | Req {session_requests}/{MAX_SESSION_REQUESTS}) Generated valid sample.", flush=True)
                else:
                    # Output failed schema validation; skip row
                    row_idx += 1
                    
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                
            except concurrent.futures.TimeoutError:
                row_retry_count += 1
                print(f"[{model_name} - Key 1] Row {row_idx} timed out (> {CALL_TIMEOUT_SEC}s) [Attempt {row_retry_count}/2].", flush=True)
                if row_retry_count >= 2:
                    print(f"[{model_name} - Key 1] Skipping slow row {row_idx}...", flush=True)
                    row_idx += 1
                    row_retry_count = 0
                time.sleep(2)
                
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    model_429_attempts += 1
                    if model_429_attempts < 3:
                        print(f"[{model_name} - Key 1] 429 burst throttle. Pausing 30s for RPM reset (Attempt {model_429_attempts}/3)...", flush=True)
                        time.sleep(30)
                        continue
                    else:
                        print(f"[{model_name} - Key 1] Daily quota genuinely exhausted (3x 429). Exiting as requested.", flush=True)
                        break
                elif "404" in err_str or "not found" in err_str.lower():
                    print(f"[{model_name} - Key 1] Model unavailable (404). Exiting.", flush=True)
                    break
                elif "504" in err_str or "deadline" in err_str.lower():
                    print(f"[{model_name} - Key 1] Server 504 Deadline. Skipping row {row_idx}...", flush=True)
                    row_idx += 1
                    time.sleep(2)
                    continue
                else:
                    print(f"[{model_name} - Key 1] Error: {err_str[:80]}... Skipping row {row_idx}.", flush=True)
                    row_idx += 1
                    time.sleep(2)
                    continue

    print(f"\nSession complete. Total samples in dataset: {total_samples} / {TARGET_SAMPLES} (Requests used: {session_requests}/{MAX_SESSION_REQUESTS})", flush=True)

if __name__ == "__main__":
    main()
