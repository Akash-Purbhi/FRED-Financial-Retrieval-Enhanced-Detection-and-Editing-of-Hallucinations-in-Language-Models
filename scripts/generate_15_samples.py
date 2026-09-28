"""
generate_15_samples.py
Generates 15 synthetic financial Q&A data samples using real FinQA financial data and Gemini.
Enforces:
1. Strict 15 RPM rate limiting (sleep 4.5s).
2. Model rotation across [gemini-3.7-flash, gemini-3.6-flash, gemini-3-flash-preview, gemini-3.8-flash].
3. Output format: JSON array of objects with keys: context, question, response, target_output.
4. Target output contains step-by-step calculations followed by tagged correction:
   <numerical><delete>wrong</delete><mark>correct</mark></numerical>
   or
   <temporal><delete>wrong</delete><mark>correct</mark></temporal>.
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

# Rotating through the requested model families
MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
]

OUTPUT_FILE = ROOT_DIR / "synthetic_finqa_cot_15.json"

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
  "context": "Clean summary or excerpt of relevant financial context from the documents",
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

def generate_sample(context_str, question_str, answer_str, model_idx):
    prompt = PROMPT_TEMPLATE.format(
        context=context_str,
        question=question_str,
        ground_truth=answer_str,
        question_escaped=question_str.replace('"', '\\"')
    )
    
    # Try starting from model_idx, rotate on failure
    for i in range(len(MODELS)):
        cur_model = MODELS[(model_idx + i) % len(MODELS)]
        try:
            model = genai.GenerativeModel(cur_model)
            response = model.generate_content(prompt)
            raw_text = clean_json_text(response.text)
            # Find json block if surrounded by extra text
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1:
                raw_text = raw_text[start:end+1]
            data = json.loads(raw_text)
            
            # preserve context
            if not data.get("context") or len(data["context"]) < 30:
                data["context"] = context_str[:600]
            data["question"] = question_str
            
            if validate_sample(data):
                return data, cur_model
            else:
                print(f"Validation failed for {cur_model}, trying next...")
        except Exception as e:
            print(f"Error with {cur_model}: {e}")
            time.sleep(2)
            continue
            
    return None, None

def main():
    load_dotenv(find_dotenv(usecwd=True), override=True)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    
    print("Loading FinQA dataset from RagBench...")
    dataset = load_dataset("rungalileo/ragbench", "finqa")
    train_data = dataset["train"]
    
    results = []
    target_count = 15
    row_idx = 0
    model_cursor = 0
    
    print(f"Starting generation of {target_count} samples rotating models with 4.5s sleep (15 RPM)...")
    
    while len(results) < target_count and row_idx < len(train_data):
        row = train_data[row_idx]
        row_idx += 1
        
        context_raw = row.get("documents") or row.get("context") or ""
        question = row.get("question") or ""
        answer = row.get("response") or row.get("answer") or ""
        
        if not context_raw or not question or not answer:
            continue
            
        context_str = format_docs(context_raw)
        
        sample, used_model = generate_sample(context_str, question, answer, model_cursor)
        model_cursor = (model_cursor + 1) % len(MODELS)
        
        # Enforce rate limit (15 RPM -> 4.5s delay)
        time.sleep(4.5)
        
        if sample:
            results.append(sample)
            print(f"[{len(results)}/{target_count}] Model: {used_model} | Q: {question[:60]}...")
            
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nDone! Saved {len(results)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
