"""
06_transform_cot_chatml.py
Transforms synthetic CoT hallucination data (synthetic_finqa_cot_5000.jsonl)
into ChatML format (chatml_finqa_cot_5000.jsonl) for Unsloth / Hugging Face SFT fine-tuning.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

INPUT_FILE = ROOT_DIR / "synthetic_finqa_cot_5000.jsonl"
OUTPUT_FILE = ROOT_DIR / "chatml_finqa_cot_5000.jsonl"


def transform_cot_to_chatml(input_path: Path, output_path: Path):
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    formatted_count = 0
    skipped_count = 0

    print(f"Reading from: {input_path.name}")
    print(f"Writing to  : {output_path.name}\n")

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                item = json.loads(line_str)
            except Exception as e:
                print(f"Skipping malformed JSON at line {line_num}: {e}")
                skipped_count += 1
                continue

            context = item.get("context", "").strip()
            question = item.get("question", "").strip()
            response = item.get("response", "").strip()
            target_output = item.get("target_output", "").strip()

            if not (context and question and response and target_output):
                skipped_count += 1
                continue

            user_content = f"Context: {context}\nQuestion: {question}\nResponse: {response}"

            chatml_record = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": target_output,
                    },
                ]
            }

            outfile.write(json.dumps(chatml_record, ensure_ascii=False) + "\n")
            formatted_count += 1

    print("Transformation Summary:")
    print(f"  Successfully formatted : {formatted_count} rows")
    print(f"  Skipped rows           : {skipped_count} rows")
    print(f"  Output saved to        : {output_path.name}\n")


def main():
    transform_cot_to_chatml(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
