"""
05_generate_cot_dataset.py
Sequential bulk generator for FRED synthetic financial CoT dataset (Target: 5,000 samples).

Strategy:
1. Sequential Exhaustion:
   - Uses Key 1 with Model 1 until daily quota (429) is reached.
   - Moves to Model 2, Model 3, etc. on Key 1.
   - When all models on Key 1 are exhausted, switches to Key 2 and repeats.
2. Verified Active Models:
   - Priority: gemini-3.1-flash-lite, gemini-3-flash-preview, gemini-3.8-flash, gemini-3.7-flash,
     gemini-3.6-flash, gemini-3.5-flash, gemini-flash-latest, gemini-flash-lite-latest, gemini-2.5-flash.
3. Connection Safety:
   - 4.5s delay between calls (strict 15 RPM free-tier compliance).
   - transport="rest" with non-blocking 35-second execution timeout.
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
RATE_LIMIT_SLEEP_SEC = 4.5
CALL_TIMEOUT_SEC = 35

MODELS = [
    "gemini-3.1-flash-lite",    # Fast, high daily quota (~1,000 req/day)
    "gemini-3-flash-preview",   # Fast, high accuracy
    "gemini-3.8-flash",         # Latest 3.8
    "gemini-3.7-flash",         # Latest 3.7
    "gemini-3.6-flash",         # Latest 3.6
    "gemini-3.5-flash",         # Latest 3.5
    "gemini-flash-latest",      # Google general Flash alias
    "gemini-flash-lite-latest", # Google general Flash-Lite alias
    "gemini-2.5-flash",         # Fallback after modern models are used up
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
        for i, d in enumerate(docs[:3]):
            d_str = str(d)
            if len(d_str) > 1000:
                d_str = d_str[:1000] + "..."
            formatted.append(f"[Doc {i+1}] {d_str}")
        return "\n\n".join(formatted)
    return str(docs)[:2500]

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
    
    # Must contain Reasoning/Calculations
    if not ("Reasoning:" in target or "calculate" in target.lower() or "=" in target or "+" in target or "-" in target or "%" in target):
        return False
    
    # Must contain tag
    tag_match = re.search(r"<(numerical|temporal)><delete>(.*?)</delete><mark>(.*?)</mark></\1>", target, re.IGNORECASE | re.DOTALL)
    if not tag_match:
        return False
    
    wrong = tag_match.group(2).strip()
    correct = tag_match.group(3).strip()
    
    if not wrong or not correct or wrong.lower() == correct.lower():
        return False
    
    # The wrong span must be in the response
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
    
    api_keys = []
    for var in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
        k = os.environ.get(var)
        if k and k.strip() and k.strip() not in api_keys:
            api_keys.append(k.strip())
            
    if not api_keys:
        raise ValueError("No GEMINI_API_KEY found in environment (.env).")
    
    print(f"Loaded {len(api_keys)} Gemini API Key(s) for sequential generation.", flush=True)
    
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
    
    current_key_idx = 0
    current_model_idx = 0
    row_idx = existing_count
    total_samples = existing_count
    model_consecutive_timeouts = 0
    
    print(f"Starting sequential generation up to {TARGET_SAMPLES} samples...")
    print(f"Active Model Pool: {MODELS}")
    print(f"Initial Key: Key {current_key_idx + 1}, Initial Model: {MODELS[current_model_idx]}", flush=True)
    
    genai.configure(api_key=api_keys[current_key_idx], transport="rest")
    current_model = genai.GenerativeModel(MODELS[current_model_idx])
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        while total_samples < TARGET_SAMPLES and row_idx < len(train_data):
            if current_key_idx >= len(api_keys):
                print("\n[INFO] All API keys and models have reached their daily quotas! Generator paused.", flush=True)
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
            
            model_name = MODELS[current_model_idx]
            key_label = f"Key {current_key_idx + 1}"
            
            try:
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
                    model_consecutive_timeouts = 0
                    print(f"[{total_samples}/{TARGET_SAMPLES}] ({model_name} [{key_label}]) Generated valid sample.", flush=True)
                else:
                    row_idx += 1
                    
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                
            except concurrent.futures.TimeoutError:
                model_consecutive_timeouts += 1
                print(f"[{model_name} - {key_label}] Request timed out (> {CALL_TIMEOUT_SEC}s) [Timeout #{model_consecutive_timeouts}].", flush=True)
                if model_consecutive_timeouts >= 2:
                    print(f"[{model_name} - {key_label}] 2 consecutive timeouts. Advancing model...", flush=True)
                    model_consecutive_timeouts = 0
                    current_model_idx += 1
                    if current_model_idx >= len(MODELS):
                        current_key_idx += 1
                        current_model_idx = 0
                        if current_key_idx < len(api_keys):
                            print(f"\n[{key_label}] All models exhausted. Switching to Key {current_key_idx + 1}...\n", flush=True)
                            genai.configure(api_key=api_keys[current_key_idx], transport="rest")
                    if current_key_idx < len(api_keys):
                        current_model = genai.GenerativeModel(MODELS[current_model_idx])
                time.sleep(2)
                
            except Exception as e:
                err_str = str(e)
                model_consecutive_timeouts = 0
                if "429" in err_str or "quota" in err_str.lower():
                    print(f"[{model_name} - {key_label}] Daily quota exhausted (429). Advancing model...", flush=True)
                    current_model_idx += 1
                elif "404" in err_str or "not found" in err_str.lower():
                    print(f"[{model_name} - {key_label}] Model unavailable (404). Advancing model...", flush=True)
                    current_model_idx += 1
                elif "504" in err_str or "deadline" in err_str.lower():
                    print(f"[{model_name} - {key_label}] Server 504 Deadline Exceeded. Advancing model...", flush=True)
                    current_model_idx += 1
                else:
                    print(f"[{model_name} - {key_label}] Error: {err_str[:80]}... Advancing model.", flush=True)
                    current_model_idx += 1
                    
                if current_model_idx >= len(MODELS):
                    current_key_idx += 1
                    current_model_idx = 0
                    if current_key_idx < len(api_keys):
                        print(f"\n[{key_label}] All models exhausted. Switching to Key {current_key_idx + 1}...\n", flush=True)
                        genai.configure(api_key=api_keys[current_key_idx], transport="rest")
                
                if current_key_idx < len(api_keys):
                    current_model = genai.GenerativeModel(MODELS[current_model_idx])
                time.sleep(2)

    print(f"\nSession complete. Total samples in dataset: {total_samples} / {TARGET_SAMPLES}", flush=True)

if __name__ == "__main__":
    main()
