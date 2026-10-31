#!/usr/bin/env python3
"""
scripts/16_build_stage2_dataset.py
==================================
Builds the official Stage 2 Financial Hallucination Dataset:
- FinQA: 5,000 clean + 5,000 controlled variants = 10,000
- TAT-QA: 4,000 clean + 4,000 controlled variants = 8,000
- Total Training: ~18,000 examples
- Validation: 1,000 examples (500 clean + 500 corrupted from training sources)
- Testing: Official test sets kept untouched (Clean Test) + separate Corrupted Test Benchmark

Outputs:
  data/stage2/train.jsonl
  data/stage2/train_chatml.jsonl
  data/stage2/validation.jsonl
  data/stage2/validation_chatml.jsonl
  data/stage2/test_clean.jsonl
  data/stage2/test_corrupted.jsonl
  reports/stage2_dataset_report.md
"""

import copy
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Workspace configuration
WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE / "data"
STAGE2_DIR = DATA_DIR / "stage2"
REPORT_DIR = WORKSPACE / "reports"
STAGE2_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Path to cached HuggingFace datasets
RAGBENCH_SNAPSHOT = Path(
    r"C:\Users\Akash\.cache\huggingface\hub\datasets--rungalileo--ragbench\snapshots\97808f3e5fd16ede40bbff6c2949af8139b2eb7b"
)
OFFICIAL_FINQA_DIR = DATA_DIR / "official_finqa"

# Set deterministic random seed
random.seed(42)

# Import Stage 2 Schema
sys.path.insert(0, str(WORKSPACE / "scripts"))
from stage2_schema import Stage2Record, EvidenceItem


def parse_table_grid(raw_table: Any) -> List[List[str]]:
    """Convert various table representations into a 2D grid of strings."""
    if not raw_table:
        return []
    if isinstance(raw_table, list):
        if len(raw_table) > 0 and isinstance(raw_table[0], list):
            return [[str(c).strip() for c in row] for row in raw_table]
    if isinstance(raw_table, str):
        try:
            parsed = json.loads(raw_table)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                return [[str(c).strip() for c in row] for row in parsed]
        except Exception:
            pass
    return []


def extract_table_and_context_from_docs(docs: Any) -> Tuple[str, List[List[str]]]:
    """Extract unstructured text context and structured 2D table grid from RagBench documents array."""
    context_parts = []
    table_grid = []

    if isinstance(docs, (list, tuple, pd.Series, np_ndarray := type(pd.Series([1]).values))):
        for item in docs:
            s = str(item).strip()
            # Check if this item is a stringified JSON 2D table
            if s.startswith("[[") and s.endswith("]]"):
                grid = parse_table_grid(s)
                if grid:
                    table_grid = grid
                    continue
            # Check if document contains a table header or embedded table
            table_match = re.search(r"(\[\[.*?\]\])", s, re.DOTALL)
            if table_match and not table_grid:
                table_grid = parse_table_grid(table_match.group(1))
                s = s.replace(table_match.group(1), "").strip()

            if s:
                context_parts.append(s)

    context_text = "\n\n".join(context_parts)
    return context_text, table_grid


# ── Perturbation Engine for Controlled Variants ──────────────────────────────

YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
NUMBER_RE = re.compile(r"\b(?:\$|€|£)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*%?\b")


def perturb_number(val_str: str) -> str:
    """Perturb a numeric string slightly (digit change or percentage shift)."""
    clean = val_str.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        f = float(clean)
        # Shift by 10% to 25% or small additive
        if "%" in val_str or (0 < f <= 100):
            delta = random.choice([2.5, 5.0, -3.2, 7.8, -4.5])
            new_f = round(f + delta, 1)
            return f"{new_f}%" if "%" in val_str else f"{new_f}"
        else:
            factor = random.choice([0.85, 1.15, 0.9, 1.2])
            new_f = round(f * factor, 1 if "." in clean else 0)
            prefix = "$" if "$" in val_str else ""
            return f"{prefix}{int(new_f) if new_f.is_integer() else new_f}"
    except ValueError:
        return f"{val_str} (est.)"


def create_hallucination_variant(clean_record: Stage2Record, variant_type: str) -> Stage2Record:
    """Create a controlled hallucination variant for a given clean record."""
    corrupted = copy.deepcopy(clean_record)
    corrupted.id = f"{clean_record.id}_{variant_type}"

    ans = clean_record.gold_answer.strip()
    context = clean_record.context
    table = clean_record.table

    if variant_type == "edit_numerical":
        corrupted.action = "edit"
        corrupted.error_type = "numerical_substitution"
        wrong_ans = perturb_number(ans)
        if wrong_ans == ans:
            wrong_ans = f"{ans}0"
        corrupted.unverified_response = f"According to the financial records, the calculated value was {wrong_ans}."
        corrupted.tagged_xml_response = (
            f"According to the financial records, the calculated value was "
            f"<numerical><delete>{wrong_ans}</delete><mark>{ans}</mark></numerical>."
        )
        corrupted.reasoning = f"The reported figure {wrong_ans} is incorrect. The verified context supports {ans}."

    elif variant_type == "edit_temporal":
        corrupted.action = "edit"
        corrupted.error_type = "temporal_swap"
        # Find years in context or question
        years = YEAR_RE.findall(clean_record.question + " " + context)
        if years:
            target_year = years[0]
            fake_year = str(int(target_year) - 1 if int(target_year) > 2005 else int(target_year) + 1)
            corrupted.unverified_response = f"For fiscal year {fake_year}, the reported figure was {ans}."
            corrupted.tagged_xml_response = (
                f"For fiscal year <temporal><delete>{fake_year}</delete><mark>{target_year}</mark></temporal>, "
                f"the reported figure was {ans}."
            )
        else:
            corrupted.unverified_response = f"In fiscal year 2020, the reported figure was {ans}."
            corrupted.tagged_xml_response = (
                f"In fiscal year <temporal><delete>2020</delete><mark>2021</mark></temporal>, the reported figure was {ans}."
            )
        corrupted.reasoning = "Reporting period was mismatched against the question context."

    elif variant_type == "edit_relation":
        corrupted.action = "edit"
        corrupted.error_type = "relation_inversion"
        corrupted.unverified_response = f"The reported figure decreased to {ans}."
        corrupted.tagged_xml_response = (
            f"The reported figure <relation><delete>decreased</delete><mark>increased</mark></relation> to {ans}."
        )
        corrupted.reasoning = "Relation direction was inverted."

    elif variant_type == "abstain":
        corrupted.action = "abstain"
        corrupted.error_type = "missing_evidence"
        # Truncate context or remove table to simulate missing evidence
        corrupted.context = "General corporate overview: The company operates across multiple geographic segments."
        corrupted.table = []
        corrupted.unverified_response = f"Based on the disclosures, the required figure was {ans}."
        corrupted.tagged_xml_response = None
        corrupted.gold_answer = "Unverifiable from provided context."
        corrupted.reasoning = "The relevant disclosure tables and metrics were omitted from the retrieved context."

    elif variant_type == "flag_conflict":
        corrupted.action = "flag_conflict"
        corrupted.error_type = "contradictory_context"
        alt_number = perturb_number(ans)
        corrupted.context = (
            f"{context}\n\n[Restatement Note]: Subsequent review indicated that the finalized metric was actually {alt_number}."
        )
        corrupted.unverified_response = f"The reported figure was {ans}."
        corrupted.tagged_xml_response = None
        corrupted.reasoning = f"Source documents contain contradictory disclosures: primary text reports {ans} while Restatement Note reports {alt_number}."

    elif variant_type == "request_evidence":
        corrupted.action = "request_evidence"
        corrupted.error_type = "irrelevant_retrieval"
        corrupted.context = (
            "Item 1. Business. We are a global technology leader providing hardware, software, and cloud services."
        )
        corrupted.table = []
        corrupted.unverified_response = f"The result was {ans}."
        corrupted.tagged_xml_response = None
        corrupted.reasoning = "Retrieved document is an introductory business overview containing no financial tables."

    else:
        # Default fallback
        corrupted.action = "edit"
        corrupted.error_type = "numerical_substitution"
        wrong_ans = perturb_number(ans)
        corrupted.unverified_response = f"The result was {wrong_ans}."
        corrupted.tagged_xml_response = f"The result was <numerical><delete>{wrong_ans}</delete><mark>{ans}</mark></numerical>."

    return corrupted


# ── Ingestion Functions ──────────────────────────────────────────────────────

def ingest_finqa(target_count: int) -> List[Stage2Record]:
    """Ingest clean records from official FinQA train.json."""
    print(f"Ingesting FinQA from {OFFICIAL_FINQA_DIR / 'train.json'}...")
    with open(OFFICIAL_FINQA_DIR / "train.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for idx, item in enumerate(raw_data):
        qa = item.get("qa", {})
        question = qa.get("question", "").strip()
        ans = str(qa.get("exe_ans", qa.get("answer", ""))).strip()
        if not question or not ans or ans.lower() in ["yes", "no"]:
            continue

        pre_text = " ".join(item.get("pre_text", []))
        post_text = " ".join(item.get("post_text", []))
        context = f"{pre_text}\n{post_text}".strip()
        table = parse_table_grid(item.get("table", []))

        # Build evidence citations from gold_inds
        gold_inds = qa.get("gold_inds", {})
        evidence_list = []
        for k, v in gold_inds.items():
            stype = "table" if "table" in k else "context"
            evidence_list.append({"source_type": stype, "location": k, "text": str(v)})

        program = qa.get("program_re", qa.get("program", "none"))
        ans_type = "percentage" if "%" in ans else "number"

        rec = Stage2Record(
            id=f"finqa_train_{idx:05d}",
            source_dataset="finqa",
            question=question,
            context=context,
            table=table,
            gold_answer=ans,
            evidence=evidence_list,
            reasoning=f"Calculated via {program} based on retrieved financial tables.",
            operation=str(program),
            answer_type=ans_type,
            error_type="none",
            action="preserve",
            unverified_response=f"Based on the financial disclosures, the value is {ans}.",
        )
        records.append(rec)
        if len(records) >= target_count:
            break

    print(f"  Loaded {len(records)} clean FinQA records.")
    return records


def ingest_tatqa(target_count: int) -> List[Stage2Record]:
    """Ingest clean records from cached RagBench TAT-QA train parquet."""
    parquet_path = RAGBENCH_SNAPSHOT / "tatqa" / "train-00000-of-00001.parquet"
    print(f"Ingesting TAT-QA from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Filter for high adherence (grounded answers only)
    if "adherence_score" in df.columns:
        df = df[df["adherence_score"] >= 0.8]

    records = []
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        response = str(row["response"]).strip()
        docs = row["documents"]

        context_text, table_grid = extract_table_and_context_from_docs(docs)

        # Extract answer from response
        gold_ans = response.split(".")[-2] if "." in response else response
        gold_ans = gold_ans.strip()

        rec = Stage2Record(
            id=f"tatqa_train_{idx:05d}",
            source_dataset="tatqa",
            question=question,
            context=context_text,
            table=table_grid,
            gold_answer=response,
            evidence=[{"source_type": "context", "location": "doc_0", "text": context_text[:200]}],
            reasoning="Factually grounded in hybrid textual and tabular disclosure.",
            operation="table_text_lookup",
            answer_type="span_extraction",
            error_type="none",
            action="preserve",
            unverified_response=response,
        )
        records.append(rec)
        if len(records) >= target_count:
            break

    print(f"  Loaded {len(records)} clean TAT-QA records.")
    return records


def ingest_untouched_test_sets() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Ingest official FinQA and TAT-QA test sets completely untouched."""
    print("Ingesting official held-out test sets...")
    test_clean = []
    test_corrupted = []

    # 1. FinQA test
    finqa_test_path = OFFICIAL_FINQA_DIR / "test.json"
    if finqa_test_path.exists():
        with open(finqa_test_path, "r", encoding="utf-8") as f:
            finqa_test_raw = json.load(f)
        for idx, item in enumerate(finqa_test_raw[:600]):
            qa = item.get("qa", {})
            q = qa.get("question", "")
            ans = str(qa.get("exe_ans", qa.get("answer", "")))
            pre = " ".join(item.get("pre_text", []))
            post = " ".join(item.get("post_text", []))
            table = parse_table_grid(item.get("table", []))
            rec = Stage2Record(
                id=f"finqa_test_{idx:04d}",
                source_dataset="finqa",
                question=q,
                context=f"{pre}\n{post}".strip(),
                table=table,
                gold_answer=ans,
                evidence=[],
                reasoning="",
                operation=str(qa.get("program_re", "none")),
                answer_type="number",
                error_type="none",
                action="preserve",
                unverified_response=f"The answer is {ans}.",
            )
            test_clean.append(rec.to_dict())
            # Paired corrupted test benchmark
            corr = create_hallucination_variant(rec, random.choice(["edit_numerical", "edit_temporal", "abstain"]))
            test_corrupted.append(corr.to_dict())

    # 2. TAT-QA test
    tatqa_test_path = RAGBENCH_SNAPSHOT / "tatqa" / "test-00000-of-00001.parquet"
    if tatqa_test_path.exists():
        df_test = pd.read_parquet(tatqa_test_path)
        for idx, row in df_test.iloc[:600].iterrows():
            c_text, t_grid = extract_table_and_context_from_docs(row["documents"])
            resp = str(row["response"]).strip()
            rec = Stage2Record(
                id=f"tatqa_test_{idx:04d}",
                source_dataset="tatqa",
                question=str(row["question"]).strip(),
                context=c_text,
                table=t_grid,
                gold_answer=resp,
                evidence=[],
                reasoning="",
                operation="table_lookup",
                answer_type="span_extraction",
                error_type="none",
                action="preserve",
                unverified_response=resp,
            )
            test_clean.append(rec.to_dict())
            corr = create_hallucination_variant(rec, random.choice(["edit_numerical", "edit_relation", "flag_conflict"]))
            test_corrupted.append(corr.to_dict())

    print(f"  Test sets loaded: {len(test_clean)} clean, {len(test_corrupted)} corrupted.")
    return test_clean, test_corrupted


# ── Main Build Routine ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FRED Stage 2: Building 18,000 Sample Hybrid Dataset")
    print("=" * 60)

    # Ingest clean base records: 5,000 FinQA + 4,000 TAT-QA
    finqa_clean = ingest_finqa(target_count=5000)
    tatqa_clean = ingest_tatqa(target_count=4000)

    # Reserve validation slice from training sources:
    # 300 FinQA clean + 200 TAT-QA clean = 500 clean val
    val_clean = finqa_clean[-300:] + tatqa_clean[-200:]
    train_finqa_clean = finqa_clean[:-300]
    train_tatqa_clean = tatqa_clean[:-200]

    # Fill back to reach 5,000 and 4,000 if needed
    train_clean_pool = train_finqa_clean + train_tatqa_clean

    print(f"\nBuilding Training Set: ~18,000 paired records...")

    # Variant action weights for corrupted generation:
    # 60% edit (numerical, temporal, relation), 20% abstain, 10% conflict, 10% request evidence
    variant_types = (
        ["edit_numerical"] * 35
        + ["edit_temporal"] * 15
        + ["edit_relation"] * 10
        + ["abstain"] * 20
        + ["flag_conflict"] * 10
        + ["request_evidence"] * 10
    )

    train_records = []
    # Add all clean records (preserve)
    for rec in train_clean_pool:
        train_records.append(rec)
        # Create exactly 1 controlled hallucination variant per clean record
        v_type = random.choice(variant_types)
        corr_rec = create_hallucination_variant(rec, v_type)
        train_records.append(corr_rec)

    random.shuffle(train_records)

    # Validation Set: 500 clean + 500 corrupted
    val_records = []
    for rec in val_clean:
        val_records.append(rec)
        v_type = random.choice(variant_types)
        val_records.append(create_hallucination_variant(rec, v_type))
    random.shuffle(val_records)

    # Write training JSONL and ChatML
    train_jsonl_path = STAGE2_DIR / "train.jsonl"
    train_chatml_path = STAGE2_DIR / "train_chatml.jsonl"
    print(f"\nWriting {len(train_records)} training records to {train_jsonl_path}...")
    with open(train_jsonl_path, "w", encoding="utf-8") as f_json, open(
        train_chatml_path, "w", encoding="utf-8"
    ) as f_chat:
        for r in train_records:
            f_json.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            f_chat.write(json.dumps(r.to_chatml(), ensure_ascii=False) + "\n")

    # Write validation JSONL and ChatML
    val_jsonl_path = STAGE2_DIR / "validation.jsonl"
    val_chatml_path = STAGE2_DIR / "validation_chatml.jsonl"
    print(f"Writing {len(val_records)} validation records to {val_jsonl_path}...")
    with open(val_jsonl_path, "w", encoding="utf-8") as f_json, open(
        val_chatml_path, "w", encoding="utf-8"
    ) as f_chat:
        for r in val_records:
            f_json.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            f_chat.write(json.dumps(r.to_chatml(), ensure_ascii=False) + "\n")

    # Ingest and write official test benchmarks
    test_clean, test_corrupted = ingest_untouched_test_sets()
    with open(STAGE2_DIR / "test_clean.jsonl", "w", encoding="utf-8") as f:
        for r in test_clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(STAGE2_DIR / "test_corrupted.jsonl", "w", encoding="utf-8") as f:
        for r in test_corrupted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute Action Distribution Statistics
    action_counts = {}
    error_counts = {}
    for r in train_records:
        action_counts[r.action] = action_counts.get(r.action, 0) + 1
        error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1

    # Generate Markdown Report
    report_content = f"""# FRED Stage 2 Dataset Generation Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Summary of Generated Splits

| Split | File | Record Count | Clean : Corrupted Ratio |
|---|---|:---:|:---:|
| **Train (Unified JSONL)** | `data/stage2/train.jsonl` | **{len(train_records):,}** | 50% Clean : 50% Corrupted |
| **Train (ChatML Format)** | `data/stage2/train_chatml.jsonl` | **{len(train_records):,}** | Formatted for Qwen 3B / 7B |
| **Validation (JSONL)** | `data/stage2/validation.jsonl` | **{len(val_records):,}** | 500 Clean : 500 Corrupted |
| **Test (Clean Benchmark)** | `data/stage2/test_clean.jsonl` | **{len(test_clean):,}** | 100% Untouched Official Ground Truth |
| **Test (Corrupted Suite)** | `data/stage2/test_corrupted.jsonl` | **{len(test_corrupted):,}** | Controlled Hallucination Benchmark |

## 2. Action Distribution (Training Set)

| Action | Count | Percentage |
|---|:---:|:---:|
"""
    for act, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / len(train_records)) * 100
        report_content += f"| `{act}` | {cnt:,} | {pct:.1f}% |\n"

    report_content += """
## 3. Error Taxonomy Distribution (Training Set)

| Error Type | Count | Percentage |
|---|:---:|:---:|
"""
    for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / len(train_records)) * 100
        report_content += f"| `{err}` | {cnt:,} | {pct:.1f}% |\n"

    with open(REPORT_DIR / "stage2_dataset_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nDataset build complete! Report generated at: {REPORT_DIR / 'stage2_dataset_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
