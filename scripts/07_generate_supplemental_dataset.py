import os
import json
import time
import datetime
from pathlib import Path
from datasets import load_dataset
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from google.generativeai.types import GenerationConfig
import concurrent.futures

WORKSPACE = Path(__file__).parent.parent
OUTPUT_FILE = WORKSPACE / "synthetic_finqa_supplemental.jsonl"

TARGETS = {
    "numerical": 500,
    "temporal": 500
}

MODEL_NAME = "gemini-3.5-flash-lite"
RATE_LIMIT_SLEEP_SEC = 4.5
CALL_TIMEOUT_SEC = 25

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
        except Exception:
            return None

def main():
    load_dotenv(find_dotenv(usecwd=True), override=True)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY not found in environment. Exiting.")
        return

    # Count existing
    counts = {k: 0 for k in TARGETS}
    seen_ids = set()
    
    # Load previously validated IDs to avoid duplicates
    VALID_FILE = WORKSPACE / "synthetic_finqa_cot_high_confidence_valid.jsonl"
    if VALID_FILE.exists():
        with open(VALID_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "source_id" in data:
                            seen_ids.add(data["source_id"])
                    except:
                        pass
    print(f"Loaded {len(seen_ids)} existing source IDs from high_confidence_valid.")

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        err_type = data.get("error_type")
                        if err_type in counts:
                            counts[err_type] += 1
                        seen_ids.add(data.get("source_id"))
                    except:
                        pass
    print(f"Resuming with counts: {counts}")


    print("Loading RagBench FinQA dataset...")
    ds = load_dataset("rungalileo/ragbench", "finqa", split="train")
    
    genai.configure(api_key=api_key, transport="rest")
    model = genai.GenerativeModel(MODEL_NAME)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:
        for row in ds:
            # Check targets
            needs_generation = [k for k, v in TARGETS.items() if counts[k] < v]
            if not needs_generation:
                print("All targets reached! Exiting.")
                break
                
            source_id = row.get("id")
            if source_id in seen_ids:
                continue
                
            # Grounding check
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
            
            response = generate_sample(model, prompt)
            time.sleep(RATE_LIMIT_SLEEP_SEC)
            
            if not response or not response.text:
                continue
                
            try:
                gen_data = json.loads(response.text)
                
                record = {
                    "source_id": source_id,
                    "source_documents": source_documents,
                    "source_response": source_response,
                    "question": question,
                    "error_type": error_type,
                    "response": gen_data.get("response", ""),
                    "target_output": gen_data.get("target_output", ""),
                    "generator_model": MODEL_NAME,
                    "generation_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                }
                
                outfile.write(json.dumps(record) + "\n")
                outfile.flush()
                counts[error_type] += 1
                seen_ids.add(source_id)
                print(f"Generated {error_type} sample. Total {error_type}: {counts[error_type]}/{TARGETS[error_type]}")
                
            except Exception as e:
                print(f"Error parsing JSON output: {e}")

if __name__ == "__main__":
    print("Generator implemented but not executed as per user request.")
    # main()
