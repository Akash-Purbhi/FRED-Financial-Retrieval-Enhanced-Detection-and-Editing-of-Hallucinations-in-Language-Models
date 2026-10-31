"""
scripts/stage2_converter.py
===========================
Exact conversion pipeline for FinQA and TAT-QA into the Unified Stage 2 Schema.

Handles:
1. Field mapping from FinQA (JSON) and TAT-QA (Parquet/RagBench).
2. Clean combination of text context and 2D markdown tables.
3. Extraction and indexing of evidence spans.
4. Preservation of numerical reasoning programs (FinQA DSL and TAT-QA explanations).
5. Normalization of answer types and mathematical operations.
6. Graceful missing-field handling and validation fallbacks.
7. SHA-256 fingerprint deduplication across question and answer pairs.
8. Extraction and serialization of ~5,000 clean FinQA and ~4,000 clean TAT-QA samples.
"""

import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Workspace configuration
WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE / "data"
STAGE2_DIR = DATA_DIR / "stage2"
STAGE2_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_FINQA_PATH = DATA_DIR / "official_finqa" / "train.json"
TATQA_PARQUET_PATH = Path(
    r"C:\Users\Akash\.cache\huggingface\hub\datasets--rungalileo--ragbench\snapshots\97808f3e5fd16ede40bbff6c2949af8139b2eb7b\tatqa\train-00000-of-00001.parquet"
)

sys.path.insert(0, str(WORKSPACE / "scripts"))
from stage2_schema import Stage2Record, EvidenceItem


# ── Normalization & Helper Utilities ─────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for hashing and comparison."""
    return re.sub(r"[^\w\s]", "", str(text).lower()).strip()


def compute_content_hash(question: str, answer: str) -> str:
    """Generate SHA-256 fingerprint for deduplication."""
    norm_q = normalize_text(question)
    norm_a = normalize_text(answer)
    return hashlib.sha256(f"{norm_q}::{norm_a}".encode("utf-8")).hexdigest()


def detect_answer_type(ans_str: str) -> str:
    """Normalize answer string to one of the 6 canonical answer types."""
    s = str(ans_str).strip()
    if not s:
        return "span_extraction"
    if "%" in s or re.search(r"\bpercent(?:age)?\b", s, re.I):
        return "percentage"
    if re.match(r"^(?:19[89]\d|20[0-3]\d)(?:-(?:0?[1-9]|1[0-2]))?$", s):
        return "date"
    if re.match(r"^(?:q[1-4]|first|second|third|fourth)\s+quarter", s, re.I):
        return "date"
    clean_num = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        float(clean_num)
        return "number"
    except ValueError:
        pass
    if len(s.split()) <= 4:
        return "entity"
    return "span_extraction"


def normalize_operation(prog_str: str) -> str:
    """Normalize operation string to canonical operator representation."""
    if not prog_str or prog_str.lower() in ("none", "", "nan"):
        return "table_text_lookup"
    p = prog_str.lower().strip()
    # Check for basic FinQA operators
    if "divide" in p and ("subtract" in p or "add" in p):
        return "growth_or_margin_formula"
    for op in ["divide", "multiply", "subtract", "add", "average", "sum"]:
        if op in p:
            return op
    return "table_lookup"


def render_table_as_markdown(table_grid: List[List[str]]) -> str:
    """Convert a 2D table grid into a clean Markdown table format."""
    if not table_grid or not isinstance(table_grid, list):
        return ""
    # Filter empty rows
    valid_rows = [[str(cell).strip() for cell in row] for row in table_grid if any(str(cell).strip() for cell in row)]
    if not valid_rows:
        return ""
    # Ensure uniform column count
    max_cols = max(len(row) for row in valid_rows)
    padded_rows = [row + [""] * (max_cols - len(row)) for row in valid_rows]

    headers = padded_rows[0]
    header_line = "| " + " | ".join(h if h else "-" for h in headers) + " |"
    sep_line = "| " + " | ".join(["---"] * max_cols) + " |"
    data_lines = ["| " + " | ".join(cell for cell in row) + " |" for row in padded_rows[1:]]

    return "\n".join([header_line, sep_line] + data_lines)


# ── 1. FinQA Conversion Logic ────────────────────────────────────────────────

def convert_finqa_record(raw_item: Dict[str, Any], record_idx: int) -> Optional[Stage2Record]:
    """
    Field Mapping: FinQA -> Stage2Record
    -----------------------------------
    - id             -> f"finqa_train_{record_idx:05d}"
    - source_dataset -> "finqa"
    - question       -> raw_item['qa']['question']
    - context        -> concatenated pre_text + post_text (excluding table)
    - table          -> raw_item['table'] (cleaned 2D array)
    - gold_answer    -> raw_item['qa']['exe_ans'] or raw_item['qa']['answer']
    - evidence       -> raw_item['qa']['gold_inds'] parsed into EvidenceItems
    - reasoning      -> steps summary and program derivation
    - operation      -> raw_item['qa']['program_re'] or raw_item['qa']['program']
    - answer_type    -> normalized via detect_answer_type()
    - error_type     -> "none" (clean base record)
    - action         -> "preserve"
    """
    qa = raw_item.get("qa")
    if not qa or not isinstance(qa, dict):
        return None

    question = str(qa.get("question", "")).strip()
    raw_ans = qa.get("exe_ans")
    if raw_ans is None or str(raw_ans).strip() == "":
        raw_ans = qa.get("answer", "")
    gold_answer = str(raw_ans).strip()

    # Reject unanswerable or boolean yes/no
    if not question or not gold_answer or gold_answer.lower() in ("yes", "no"):
        return None

    # Context assembly: pre_text + post_text
    pre_text = " ".join(raw_item.get("pre_text", [])).strip()
    post_text = " ".join(raw_item.get("post_text", [])).strip()
    context = f"{pre_text}\n\n{post_text}".strip() if pre_text and post_text else (pre_text or post_text)

    # Table grid
    table_grid = raw_item.get("table", [])
    if not isinstance(table_grid, list):
        table_grid = []

    # Evidence extraction from gold_inds
    gold_inds = qa.get("gold_inds", {})
    evidence_list = []
    if isinstance(gold_inds, dict):
        for loc_key, val_text in gold_inds.items():
            source_type = "table" if "table" in str(loc_key).lower() else "context"
            evidence_list.append(
                {"source_type": source_type, "location": str(loc_key), "text": str(val_text).strip()}
            )

    # Numerical reasoning program preservation
    prog = qa.get("program_re") or qa.get("program") or "none"
    steps = qa.get("steps", [])
    step_descriptions = []
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict) and "op" in s:
                step_descriptions.append(f"{s.get('op')}({s.get('arg1')}, {s.get('arg2')}) -> {s.get('res')}")

    reasoning_text = (
        f"Derived via {prog}. Steps: {'; '.join(step_descriptions)}."
        if step_descriptions
        else f"Derived via formula: {prog}."
    )

    ans_type = detect_answer_type(gold_answer)
    norm_op = normalize_operation(prog)

    record = Stage2Record(
        id=f"finqa_train_{record_idx:05d}",
        source_dataset="finqa",
        question=question,
        context=context,
        table=table_grid,
        gold_answer=gold_answer,
        evidence=evidence_list,
        reasoning=reasoning_text,
        operation=norm_op,
        answer_type=ans_type,
        error_type="none",
        action="preserve",
        unverified_response=f"According to the financial records, {question.rstrip('?')} is {gold_answer}.",
    )
    is_valid, _ = record.validate()
    return record if is_valid else None


# ── 2. TAT-QA Conversion Logic ───────────────────────────────────────────────

def convert_tatqa_record(row: pd.Series, record_idx: int) -> Optional[Stage2Record]:
    """
    Field Mapping: TAT-QA (RagBench Parquet) -> Stage2Record
    --------------------------------------------------------
    - id             -> f"tatqa_train_{record_idx:05d}"
    - source_dataset -> "tatqa"
    - question       -> row['question']
    - context        -> extracted text passages from row['documents']
    - table          -> extracted 2D JSON grid from row['documents']
    - gold_answer    -> row['response']
    - evidence       -> parsed from row['sentence_support_information']
    - reasoning      -> explanation from sentence_support_information
    - operation      -> detected from math expressions in explanation
    - answer_type    -> normalized via detect_answer_type()
    - error_type     -> "none"
    - action         -> "preserve"
    """
    question = str(row.get("question", "")).strip()
    response = str(row.get("response", "")).strip()
    if not question or not response:
        return None

    # Documents separation into text context and 2D table
    docs = row.get("documents")
    context_chunks = []
    table_grid = []

    if isinstance(docs, (list, tuple, pd.Series, np_ndarray := type(pd.Series([1]).values))):
        for item in docs:
            s = str(item).strip()
            # Detect JSON table grid
            if s.startswith("[[") and s.endswith("]]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                        table_grid = parsed
                        continue
                except Exception:
                    pass
            # Check embedded table match
            match = re.search(r"(\[\[.*?\]\])", s, re.DOTALL)
            if match and not table_grid:
                try:
                    parsed = json.loads(match.group(1))
                    if isinstance(parsed, list) and isinstance(parsed[0], list):
                        table_grid = parsed
                        s = s.replace(match.group(1), "").strip()
                except Exception:
                    pass
            if s:
                context_chunks.append(s)

    context = "\n\n".join(context_chunks).strip()

    # Evidence and reasoning from sentence_support_information
    evidence_list = []
    reasoning_parts = []
    support_info = row.get("sentence_support_information")
    if isinstance(support_info, (list, np_ndarray := type(pd.Series([1]).values))):
        for s_idx, item in enumerate(support_info):
            if isinstance(item, dict):
                expl = item.get("explanation", "")
                if expl:
                    reasoning_parts.append(expl)
                supp_keys = item.get("supporting_sentence_keys", [])
                for sk in (supp_keys if isinstance(supp_keys, (list, np_ndarray)) else [supp_keys]):
                    evidence_list.append(
                        {"source_type": "context", "location": str(sk), "text": expl[:120]}
                    )

    reasoning_text = " ".join(reasoning_parts) if reasoning_parts else "Directly extracted from hybrid context."
    operation_type = normalize_operation(reasoning_text)
    ans_type = detect_answer_type(response)

    record = Stage2Record(
        id=f"tatqa_train_{record_idx:05d}",
        source_dataset="tatqa",
        question=question,
        context=context,
        table=table_grid,
        gold_answer=response,
        evidence=evidence_list,
        reasoning=reasoning_text,
        operation=operation_type,
        answer_type=ans_type,
        error_type="none",
        action="preserve",
        unverified_response=response,
    )
    is_valid, _ = record.validate()
    return record if is_valid else None


# ── 3. Deduplication and Batch Conversion Pipeline ───────────────────────────

def run_conversion_pipeline(
    finqa_target: int = 5000, tatqa_target: int = 4000
) -> Tuple[List[Stage2Record], List[Stage2Record]]:
    """Execute clean, deduplicated extraction for both FinQA and TAT-QA."""
    print("=" * 65)
    print("FRED Stage 2: FinQA & TAT-QA Conversion Pipeline")
    print("=" * 65)

    seen_hashes: Set[str] = set()

    # 1. Convert FinQA
    print(f"Ingesting FinQA from: {OFFICIAL_FINQA_PATH}")
    with open(OFFICIAL_FINQA_PATH, "r", encoding="utf-8") as f:
        finqa_raw = json.load(f)

    finqa_clean_records: List[Stage2Record] = []
    duplicate_finqa_count = 0

    for idx, raw_item in enumerate(finqa_raw):
        record = convert_finqa_record(raw_item, len(finqa_clean_records) + 1)
        if not record:
            continue

        h = compute_content_hash(record.question, record.gold_answer)
        if h in seen_hashes:
            duplicate_finqa_count += 1
            continue
        seen_hashes.add(h)

        finqa_clean_records.append(record)
        if len(finqa_clean_records) >= finqa_target:
            break

    print(f"  FinQA: Successfully converted {len(finqa_clean_records):,} clean records (Duplicates filtered: {duplicate_finqa_count})")

    # 2. Convert TAT-QA
    print(f"\nIngesting TAT-QA from: {TATQA_PARQUET_PATH}")
    tatqa_df = pd.read_parquet(TATQA_PARQUET_PATH)
    tatqa_clean_records: List[Stage2Record] = []
    duplicate_tatqa_count = 0

    for idx, (_, row) in enumerate(tatqa_df.iterrows()):
        record = convert_tatqa_record(row, len(tatqa_clean_records) + 1)
        if not record:
            continue

        h = compute_content_hash(record.question, record.gold_answer)
        if h in seen_hashes:
            duplicate_tatqa_count += 1
            continue
        seen_hashes.add(h)

        tatqa_clean_records.append(record)
        if len(tatqa_clean_records) >= tatqa_target:
            break

    print(f"  TAT-QA: Successfully converted {len(tatqa_clean_records):,} clean records (Duplicates filtered: {duplicate_tatqa_count})")

    # 3. Save JSONL outputs
    out_finqa = STAGE2_DIR / "clean_finqa_5000.jsonl"
    out_tatqa = STAGE2_DIR / "clean_tatqa_4000.jsonl"

    print(f"\nSaving clean FinQA to: {out_finqa}")
    with open(out_finqa, "w", encoding="utf-8") as f:
        for r in finqa_clean_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saving clean TAT-QA to: {out_tatqa}")
    with open(out_tatqa, "w", encoding="utf-8") as f:
        for r in tatqa_clean_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    return finqa_clean_records, tatqa_clean_records


if __name__ == "__main__":
    finqa_records, tatqa_records = run_conversion_pipeline(finqa_target=5000, tatqa_target=4000)

    print("\n" + "=" * 65)
    print("SAMPLE CONVERTED RECORD: FinQA")
    print("=" * 65)
    sample_finqa = finqa_records[0].to_dict()
    print(json.dumps(sample_finqa, indent=2, ensure_ascii=False))

    print("\n" + "=" * 65)
    print("SAMPLE CONVERTED RECORD: TAT-QA")
    print("=" * 65)
    sample_tatqa = tatqa_records[0].to_dict()
    print(json.dumps(sample_tatqa, indent=2, ensure_ascii=False))
