"""Summarize assistant actions in a ChatML JSONL dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "stage2" / "train_chatml.jsonl"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count assistant actions and print selected action examples."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help=f"ChatML JSONL dataset (default: {DEFAULT_TRAIN_PATH})",
    )
    parser.add_argument(
        "--edit-examples",
        type=int,
        default=3,
        help="Number of edit examples to print (default: 3)",
    )
    parser.add_argument(
        "--delete-examples",
        type=int,
        default=3,
        help="Number of delete examples to print (default: 3)",
    )
    return parser.parse_args()


def analyze_dataset(
    dataset_path: Path,
    *,
    edit_example_limit: int,
    delete_example_limit: int,
) -> tuple[Counter[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return action counts, delete examples, and edit examples."""
    action_counts: Counter[str] = Counter()
    delete_examples: list[dict[str, Any]] = []
    edit_examples: list[dict[str, Any]] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                messages = record["messages"]
            except (json.JSONDecodeError, KeyError, TypeError):
                action_counts["invalid_record"] += 1
                continue

            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue

                try:
                    output = json.loads(message["content"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    action_counts["invalid_json"] += 1
                    continue

                if not isinstance(output, dict):
                    action_counts["invalid_json"] += 1
                    continue

                action = output.get("action", "missing")
                action_counts[action] += 1

                if action == "delete" and len(delete_examples) < delete_example_limit:
                    delete_examples.append(output)
                elif action == "edit" and len(edit_examples) < edit_example_limit:
                    edit_examples.append(output)

    return action_counts, delete_examples, edit_examples


def main() -> None:
    arguments = parse_arguments()
    if arguments.edit_examples < 0 or arguments.delete_examples < 0:
        raise SystemExit("Example limits must be non-negative.")
    if not arguments.dataset.is_file():
        raise SystemExit(f"Dataset file not found: {arguments.dataset}")

    action_counts, delete_examples, edit_examples = analyze_dataset(
        arguments.dataset,
        edit_example_limit=arguments.edit_examples,
        delete_example_limit=arguments.delete_examples,
    )

    print(f"DATASET: {arguments.dataset}")
    print("\nACTION COUNTS:")
    print(action_counts)

    print("\nDELETE ACTION EXAMPLES:")
    for example in delete_examples:
        print(json.dumps(example, ensure_ascii=False))

    print("\nEDIT EXAMPLES:")
    for example in edit_examples:
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print("-" * 80)


if __name__ == "__main__":
    main()
