"""
05_generate_cot_dataset.py
Production-grade bulk generator for FRED synthetic financial CoT dataset (Target: 5,000 samples).

Features:
1. Multi-Model Rotation: Rotates across Gemini 3.7 Flash, 3.6 Flash, 3 Flash Preview, and 3.8 Flash.
2. Quota & Rate Limit Protection:
   - 4.5s delay between calls (strict 15 RPM compliance).
   - Dynamic failover: Automatically catches HTTP 429 / ResourceExhausted and rotates to the next model.
3. Quality & Formatting Enforcement:
   - Deliberate numerical or temporal error in `response`.
   - Step-by-step mathematical reasoning in `target_output`.
   - Proper XML tagging: <numerical><delete>wrong</delete><mark>correct</mark></numerical>
     or <temporal><delete>wrong</delete><mark>correct</mark></temporal>.
4. Auto-Resume & Fault Tolerance:
   - Reads existing valid lines from synthetic_finqa_cot_5000.jsonl.
   - Appends line-by-line with flush, preventing data loss on interruption.
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Configurable constants
OUTPUT_FILE = ROOT_DIR / "synthetic_finqa_cot_5000.jsonl"
TARGET_SAMPLES = 5000
RATE_LIMIT_SLEEP_SEC = 4.5
CONSECUTIVE_FAILURE_LIMIT = 5

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
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

API_KEYS = []
KEY_MODEL_COOLDOWNS = {}

def is_pair_available(model_name, key_idx):
    return time.time() >= KEY_MODEL_COOLDOWNS.get((model_name, key_idx), 0)

def generate_sample_with_rotation(context_str, question_str, answer_str, current_model_idx, current_key_idx):
    prompt = PROMPT_TEMPLATE.format(
        context=context_str,
        question=question_str,
        ground_truth=answer_str,
        question_escaped=question_str.replace('"', '\\"')
    )
    
    total_pairs = len(MODELS) * len(API_KEYS)
    start_offset = current_model_idx * len(API_KEYS) + current_key_idx
    
    for attempt in range(total_pairs):
        pair_idx = (start_offset + attempt) % total_pairs
        model_idx = pair_idx // len(API_KEYS)
        key_idx = pair_idx % len(API_KEYS)
        
        model_name = MODELS[model_idx]
        api_key = API_KEYS[key_idx]
        
        if not is_pair_available(model_name, key_idx):
            continue
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw_text = clean_json_text(response.text)
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                raw_text = raw_text[start:end+1]
            data = json.loads(raw_text, strict=False)
            
            if not data.get("context") or len(data["context"]) < 30:
                data["context"] = context_str[:600]
            data["question"] = question_str
            
            if validate_sample(data):
                return data, f"{model_name} [Key {key_idx+1}]", model_idx, key_idx
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                print(f"[{model_name} - Key {key_idx+1}] Quota limit. Cooldown (30 min).", flush=True)
                KEY_MODEL_COOLDOWNS[(model_name, key_idx)] = time.time() + 1800
            elif "404" in err_str or "no longer available" in err_str.lower():
                print(f"[{model_name} - Key {key_idx+1}] Discontinued. Skipping.", flush=True)
                KEY_MODEL_COOLDOWNS[(model_name, key_idx)] = time.time() + 86400 * 365
            else:
                print(f"[{model_name} - Key {key_idx+1}] Warning: {err_str[:80]}... Rotating.", flush=True)
            time.sleep(1)
            continue
            
    return None, None, current_model_idx, current_key_idx

def main():
    global API_KEYS
    load_dotenv(find_dotenv(usecwd=True), override=True)
    
    # Load all available keys from .env
    for var in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"]:
        k = os.environ.get(var)
        if k and k.strip() and k.strip() not in API_KEYS:
            API_KEYS.append(k.strip())
            
    if not API_KEYS:
        raise ValueError("No GEMINI_API_KEY found in environment.")
    print(f"Loaded {len(API_KEYS)} Gemini API Key(s) for multi-key rotation.", flush=True)
    
    # Count existing rows
    existing_samples = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_samples.append(json.loads(line))
                    except Exception:
                        pass
        print(f"Resuming: found {len(existing_samples)} existing rows in {OUTPUT_FILE.name}")
    
    if len(existing_samples) >= TARGET_SAMPLES:
        print(f"Target of {TARGET_SAMPLES} already reached! Exiting.")
        return

    print("Loading FinQA dataset from RagBench...")
    dataset = load_dataset("rungalileo/ragbench", "finqa")
    train_data = dataset["train"]
    
    row_idx = len(existing_samples)
    model_idx = 0
    key_idx = 0
    consecutive_fails = 0
    total_added = 0
    
    print(f"Starting bulk generation up to {TARGET_SAMPLES} samples...")
    print(f"Active model pool: {MODELS}")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        while (len(existing_samples) + total_added) < TARGET_SAMPLES and row_idx < len(train_data):
            row = train_data[row_idx]
            row_idx += 1
            
            context_raw = row.get("documents") or row.get("context") or ""
            question = row.get("question") or ""
            answer = row.get("response") or row.get("answer") or ""
            
            if not context_raw or not question or not answer:
                continue
                
            context_str = format_docs(context_raw)
            sample, used_label, next_model_idx, next_key_idx = generate_sample_with_rotation(
                context_str, question, answer, model_idx, key_idx
            )
            
            # Enforce strict 15 RPM
            time.sleep(RATE_LIMIT_SLEEP_SEC)
            
            if sample:
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
                outfile.flush()
                total_added += 1
                consecutive_fails = 0
                key_idx = (next_key_idx + 1) % len(API_KEYS)
                if key_idx == 0:
                    model_idx = (next_model_idx + 1) % len(MODELS)
                else:
                    model_idx = next_model_idx
                current_total = len(existing_samples) + total_added
                print(f"[{current_total}/{TARGET_SAMPLES}] ({used_label}) Generated valid sample.", flush=True)
            else:
                consecutive_fails += 1
                key_idx = (key_idx + 1) % len(API_KEYS)
                if key_idx == 0:
                    model_idx = (model_idx + 1) % len(MODELS)
                if consecutive_fails >= CONSECUTIVE_FAILURE_LIMIT * len(MODELS) * len(API_KEYS):
                    print(f"Too many consecutive failures across all models and keys. Pausing.", flush=True)
                    break

    print(f"\nFinished session. Total samples in dataset: {len(existing_samples) + total_added}", flush=True)

if __name__ == "__main__":
    main()
