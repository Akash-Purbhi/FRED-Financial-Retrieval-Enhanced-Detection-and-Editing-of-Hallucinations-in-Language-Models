"""
04_transform_chatml.py
Transforms synthetic hallucination data into ChatML format for Unsloth fine-tuning.
Creates erroneous input passage and target correction labels with swapped delete/mark spans.
"""

import sys
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

INPUT_FILE = ROOT_DIR / "synthetic_finqa_1500.jsonl"
OUTPUT_FILE = ROOT_DIR / "chatml_finqa_1500.jsonl"

VALID_TAGS = "temporal|numerical|entity|relation|contradictory|unverifiable"


def create_erroneous_passage(tagged_response: str) -> str:
    """
    Create the input erroneous passage:
      - Completely remove <delete> tags AND their inner text.
      - Remove <mark> and </mark> tags but KEEP their inner text.
      - Strip all outer error tags (e.g. <numerical>, </numerical>).
    """
    # 1. Remove <delete>...</delete> along with its inner content
    text = re.sub(r"<delete>[\s\S]*?</delete>", "", tagged_response, flags=re.IGNORECASE)

    # 2. Remove <mark> and </mark> tags, preserving inner text
    text = re.sub(r"</?mark>", "", text, flags=re.IGNORECASE)

    # 3. Strip outer XML error tags
    text = re.sub(rf"</?(?:{VALID_TAGS})>", "", text, flags=re.IGNORECASE)

    return text


def create_target_output(tagged_response: str) -> str:
    """
    Create the assistant target output label:
      - Keep outer XML error tags.
      - Swap contents: what was in <delete> moves to <mark>,
        and what was in <mark> moves to <delete>.
    """
    pattern = re.compile(
        rf"<({VALID_TAGS})>(.*?)</\1>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def swap_tags(match):
        tag_name = match.group(1).lower()
        inner = match.group(2)

        del_match = re.search(r"<delete>(.*?)</delete>", inner, flags=re.IGNORECASE | re.DOTALL)
        mark_match = re.search(r"<mark>(.*?)</mark>", inner, flags=re.IGNORECASE | re.DOTALL)

        if del_match and mark_match:
            orig_del = del_match.group(1)
            orig_mark = mark_match.group(1)
            return f"<{tag_name}><delete>{orig_mark}</delete><mark>{orig_del}</mark></{tag_name}>"

        return match.group(0)

    return pattern.sub(swap_tags, tagged_response)


def format_documents(docs) -> str:
    """Format documents list or string for ChatML prompt context."""
    if isinstance(docs, list):
        return "\n\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(docs))
    return str(docs)


def transform_to_chatml(input_path: Path, output_path: Path):
    """
    Reads synthetic dataset line by line and converts to ChatML format.
    """
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

            documents = item.get("documents", "")
            question = item.get("question", "")
            tagged_response = item.get("tagged_response", "")

            if not tagged_response:
                skipped_count += 1
                continue

            erroneous_passage = create_erroneous_passage(tagged_response)
            target_output = create_target_output(tagged_response)
            formatted_docs = format_documents(documents)

            user_content = f"Context: {formatted_docs}\nQuestion: {question}\nResponse: {erroneous_passage}"

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
    transform_to_chatml(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
