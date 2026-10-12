import os
import sys
import json
import time
import datetime
import re
import traceback
import concurrent.futures
from collections import Counter
from pathlib import Path
from datasets import load_dataset
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv, find_dotenv

WORKSPACE = Path(__file__).parent.parent
DATA_DIR = WORKSPACE / "data" / "supplemental"
REPORT_DIR = WORKSPACE / "reports"

RAW_FILE = DATA_DIR / "supplemental_raw.jsonl"
VALID_FILE = DATA_DIR / "supplemental_valid.jsonl"
REJECTED_FILE = DATA_DIR / "supplemental_rejected.jsonl"
REPORT_FILE = REPORT_DIR / "supplemental_generation_report.md"
HIGH_CONF_VALID_FILE = WORKSPACE / "synthetic_finqa_cot_high_confidence_valid.jsonl"

TARGETS = {
    "numerical": 700,
    "temporal": 500
}

# Added the models as instructed, assuming 3.1 was a typo for 3.5
MODELS_TO_USE = [
    {"key_env": "GEMINI_API_KEY_4", "model_name": "gemini-3.5-flash-lite"},
    {"key_env": "GEMINI_API_KEY_3", "model_name": "gemini-3.5-flash-lite"} 
]

RATE_LIMIT_SLEEP_SEC = 2.0
CALL_TIMEOUT_SEC = 30

PROMPT_TEMPLATE = """\
You are an expert data generator for a financial question-answering dataset.
Your task is to introduce exactly ONE hallucination of type `{error_type}` into the provided source response.

You must provide a CoT reasoning block and the final outputs.
Your output MUST be a valid JSON object matching the following structure exactly:
{{
  "error_type": "{error_type}",
  "response": "<insert the complete corrupted response here>",
  "target_output": "Reasoning: <explain the calculation or reasoning based on the source documents, explain the error, and provide the correction>. Correction: <rewrite the FULL corrupted response verbatim, but replace the hallucinated span using exact XML tags <{error_type}><delete>WRONG_SPAN</delete><mark>CORRECT_SPAN</mark></{error_type}>>"
}}

Rules:
1. The `response` field is the complete corrupted answer.
2. The `target_output` must contain:
   - a Reasoning section;
   - `Correction:`;
   - the ENTIRE corrupted response copied verbatim;
   - exactly one replacement expressed with XML tags.
3. The text after `Correction:` must be the full corrupted response. It must not be only the XML tag.
4. Preserve every character, word, sentence, number, and punctuation mark outside the edited span.
5. Do not summarize, shorten, paraphrase, or rewrite the response.
6. Make exactly ONE change in the source response.
7. The change must be of type {error_type}.
8. Require exactly:
   - one outer `<numerical>` or `<temporal>` tag;
   - one `<delete>` block;
   - one `<mark>` block;
   - no additional XML tags.
9. Output ONLY valid JSON. No markdown backticks.

Example:
Response:
"The company reported revenue of $80 million in 2021 and $90 million in 2022."

Correct target:
"Reasoning: The context supports the reported values. Correction: The company reported revenue of <numerical><delete>$80 million</delete><mark>$85 million</mark></numerical> in 2021 and $90 million in 2022."

Incorrect target:
"Reasoning: The value is wrong. Correction: <numerical><delete>$80 million</delete><mark>$85 million</mark></numerical>"

Source Documents:
{source_documents}

Question: {question}

Source Response: {source_response}
"""

def validate_row(data):
    reasons = []
    required = ["source_id", "source_documents", "source_response", "question", "error_type", "response", "target_output"]
    for req in required:
        if not data.get(req):
            reasons.append(f"missing_{req}")
    if reasons: return reasons
    
    target = data["target_output"]
    response = data["response"]
    error_type = data["error_type"]
    
    outer_tags = re.findall(rf'<{error_type}>', target.lower())
    outer_closes = re.findall(rf'</{error_type}>', target.lower())
    if len(outer_tags) != 1 or len(outer_closes) != 1:
        reasons.append("not_exactly_one_outer_tag")
        
    deletes = re.findall(r'<delete>(.*?)</delete>', target, flags=re.DOTALL)
    if len(deletes) != 1:
        reasons.append("not_exactly_one_delete_span")
        
    marks = re.findall(r'<mark>(.*?)</mark>', target, flags=re.DOTALL)
    if len(marks) != 1:
        reasons.append("not_exactly_one_mark_span")
        
    if len(deletes) == 1 and len(marks) == 1 and len(outer_tags) == 1:
        del_text = deletes[0]
        mark_text = marks[0]
        
        if del_text not in response:
            reasons.append("wrong_span_not_in_response")
            
        if del_text == mark_text:
            reasons.append("wrong_and_corrected_spans_identical")
            
        idx = target.find("Correction:")
        if idx != -1:
            corr_text = target[idx + len("Correction:"):].strip()
            
            tag_block_match = re.search(rf'<{error_type}>.*?</{error_type}>', corr_text, flags=re.IGNORECASE|re.DOTALL)
            if tag_block_match:
                tag_block = tag_block_match.group(0)
                reconstructed = corr_text.replace(tag_block, del_text)
                if reconstructed.strip().rstrip('.') != response.strip().rstrip('.'):
                    reasons.append("unrelated_text_changed_or_bad_replacement")
            else:
                reasons.append("malformed_outer_tag_block")
        else:
            reasons.append("missing_correction_section")
            
    return reasons

def generate_sample(model, prompt):
    def _call():
        return model.generate_content(
            prompt,
            generation_config=GenerationConfig(response_mime_type="application/json")
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            return future.result(timeout=CALL_TIMEOUT_SEC)
        except Exception as e:
            return e

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(find_dotenv(usecwd=True), override=True)
    
    seen_ids = set()
    valid_counts = {"numerical": 0, "temporal": 0}
    
    # 1. Load existing valid source IDs from high_confidence_valid
    if HIGH_CONF_VALID_FILE.exists():
        with open(HIGH_CONF_VALID_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if "source_id" in data:
                        seen_ids.add(data["source_id"])
                except:
                    pass
    print(f"Loaded {len(seen_ids)} IDs from high_confidence_valid")
    
    # 2. Load existing generated IDs and valid counts from current run
    if VALID_FILE.exists():
        with open(VALID_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    seen_ids.add(data["source_id"])
                    err_type = data.get("error_type")
                    if err_type in valid_counts:
                        valid_counts[err_type] += 1
                except:
                    pass
                    
    if REJECTED_FILE.exists():
        with open(REJECTED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    seen_ids.add(data["source_id"])
                except:
                    pass
                    
    print(f"Resuming with valid counts: {valid_counts}")
    
    ds = load_dataset("rungalileo/ragbench", "finqa", split="train")
    
    stats = {
        "api_attempts": 0,
        "api_failures": 0,
        "json_errors": 0,
        "valid_rows": Counter(),
        "rejected_rows": Counter(),
        "rejection_reasons": Counter(),
        "exact_replace_fails": 0,
        "total_parsed": 0,
        "models_used": set(),
        "keys_used": set()
    }
    
    current_model_idx = 0
    
    def setup_model():
        if current_model_idx >= len(MODELS_TO_USE):
            print("All models/keys exhausted.")
            return None
        cfg = MODELS_TO_USE[current_model_idx]
        key = os.environ.get(cfg["key_env"], "").strip()
        if not key:
            print(f"Key {cfg['key_env']} not found.")
            return None
        genai.configure(api_key=key, transport="rest")
        model = genai.GenerativeModel(cfg["model_name"])
        stats["models_used"].add(cfg["model_name"])
        stats["keys_used"].add(cfg["key_env"])
        return model

    model = setup_model()
    if not model:
        print("Could not setup initial model.")
        return
        
    with open(RAW_FILE, "a", encoding="utf-8") as raw_f, \
         open(VALID_FILE, "a", encoding="utf-8") as val_f, \
         open(REJECTED_FILE, "a", encoding="utf-8") as rej_f:
             
        for row in ds:
            needs_generation = [k for k, v in TARGETS.items() if valid_counts[k] < v]
            if not needs_generation:
                print("All targets reached!")
                break
                
            source_id = row.get("id")
            if source_id in seen_ids:
                continue
                
            if row.get("adherence_score", 1.0) != 1.0 or row.get("unsupported_response_sentence_keys"):
                continue
                
            docs = row.get("documents", [])
            source_documents = json.dumps(docs) if isinstance(docs, list) else str(docs)
            question = row.get("question", "")
            source_response = row.get("response") or row.get("answer") or ""
            
            if not source_documents or not question or not source_response:
                continue
                
            error_type = needs_generation[0]
            prompt = PROMPT_TEMPLATE.format(
                error_type=error_type,
                source_documents=source_documents,
                question=question,
                source_response=source_response
            )
            
            stats["api_attempts"] += 1
            if stats["api_attempts"] % 50 == 0:
                print(f"[{valid_counts['numerical']}/{TARGETS['numerical']} N | {valid_counts['temporal']}/{TARGETS['temporal']} T] Attempt {stats['api_attempts']}")
            
            response = generate_sample(model, prompt)
            
            is_exhausted = False
            if isinstance(response, Exception):
                err_str = str(response).lower()
                if "429" in err_str or "exhausted" in err_str or "quota" in err_str:
                    is_exhausted = True
                else:
                    print(f"API Failure: {response}")
                    stats["api_failures"] += 1
                    time.sleep(RATE_LIMIT_SLEEP_SEC)
                    continue
                    
            if is_exhausted:
                print("Quota exhausted! Switching to next key/model...")
                current_model_idx += 1
                model = setup_model()
                if not model:
                    break
                continue
                
            if not response or not hasattr(response, "text"):
                stats["api_failures"] += 1
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                continue
                
            try:
                gen_data = json.loads(response.text)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                continue
                
            stats["total_parsed"] += 1
            record = {
                "source_id": source_id,
                "source_documents": source_documents,
                "source_response": source_response,
                "question": question,
                "error_type": error_type,
                "response": gen_data.get("response", ""),
                "target_output": gen_data.get("target_output", ""),
                "generator_model": MODELS_TO_USE[current_model_idx]["model_name"],
                "generation_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
            }
            
            raw_f.write(json.dumps(record) + "\n")
            raw_f.flush()
            
            reasons = validate_row(record)
            if not reasons:
                val_f.write(json.dumps(record) + "\n")
                val_f.flush()
                valid_counts[error_type] += 1
                stats["valid_rows"][error_type] += 1
            else:
                record["rejection_reasons"] = reasons
                rej_f.write(json.dumps(record) + "\n")
                rej_f.flush()
                stats["rejected_rows"][error_type] += 1
                stats["rejection_reasons"][", ".join(reasons)] += 1
                if "unrelated_text_changed_or_bad_replacement" in reasons:
                    stats["exact_replace_fails"] += 1
                    
            seen_ids.add(source_id)
            time.sleep(RATE_LIMIT_SLEEP_SEC)
            
    # Generate report
    total_valid = sum(stats["valid_rows"].values())
    total_rejected = sum(stats["rejected_rows"].values())
    valid_rate = (total_valid / stats["total_parsed"]) * 100 if stats["total_parsed"] > 0 else 0
    exact_success_rate = ((stats["total_parsed"] - stats["exact_replace_fails"]) / stats["total_parsed"]) * 100 if stats["total_parsed"] > 0 else 0
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Supplemental Generation Report\n\n")
        f.write(f"**API Requests Attempted**: {stats['api_attempts']}\n")
        f.write(f"**API Failures/Timeouts**: {stats['api_failures']}\n")
        f.write(f"**Malformed/Incomplete JSON**: {stats['json_errors']}\n\n")
        
        f.write(f"**Valid Rows**: {total_valid}\n")
        f.write(f"- Numerical: {stats['valid_rows']['numerical']}\n")
        f.write(f"- Temporal: {stats['valid_rows']['temporal']}\n\n")
        
        f.write(f"**Rejected Rows**: {total_rejected}\n")
        f.write(f"- Numerical: {stats['rejected_rows']['numerical']}\n")
        f.write(f"- Temporal: {stats['rejected_rows']['temporal']}\n\n")
        
        f.write(f"**Final Valid Rate**: {valid_rate:.1f}%\n")
        f.write(f"**Exact-Replacement Success Rate**: {exact_success_rate:.1f}%\n\n")
        
        f.write("## Rejection Reasons\n")
        for r, c in stats["rejection_reasons"].items():
            f.write(f"- {r}: {c}\n")
            
        f.write("\n## Keys & Models Used\n")
        for k in stats["keys_used"]:
            f.write(f"- {k}\n")
        for m in stats["models_used"]:
            f.write(f"- {m}\n")
            
    print("Bulk generation finished.")

if __name__ == "__main__":
    main()
