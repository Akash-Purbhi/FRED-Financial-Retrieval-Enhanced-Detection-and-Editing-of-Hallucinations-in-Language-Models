#!/usr/bin/env python3
"""
11_validate_hybrid_dataset.py
==============================
Post-build validation of the final hybrid FRED dataset.

Validates every record against 13 checks:
  1.  Valid JSON
  2.  Exactly one error type for corrupted examples
  3.  Exactly one outer XML tag
  4.  Exactly one delete block
  5.  Exactly one mark block
  6.  Delete span appears in the corrupted response
  7.  Mark span is the canonical correct value
  8.  Delete and mark differ
  9.  Untagged target equals the corrected response
  10. Exactly one textual replacement
  11. No residual wrong span at the edited location
  12. Clean examples contain no XML tags
  13. No source leakage across splits

Reads:
  data/final/{train,validation,test}.jsonl

Outputs:
  Validation summary to stdout and appends to reports/final_dataset_report.md
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

WORKSPACE  = Path(__file__).resolve().parent.parent
FINAL_DIR  = WORKSPACE / "data" / "final"
REPORT_OUT = WORKSPACE / "reports" / "final_dataset_report.md"

SPLITS = ["train", "validation", "test"]

XML_TAG_RE    = re.compile(r"<(numerical|temporal)>", re.I)
DELETE_RE     = re.compile(r"<delete>(.*?)</delete>", re.DOTALL)
MARK_RE       = re.compile(r"<mark>(.*?)</mark>", re.DOTALL)
OUTER_BLOCK   = re.compile(r"<(numerical|temporal)>.*?</\1>", re.I | re.DOTALL)


def norm_number(val: str) -> str:
    v = val.strip().replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return f"{float(v):g}"
    except ValueError:
        return v.lower().strip()


def validate_record(record: dict, idx: int, split_name: str) -> list:
    """Validate a single record. Returns list of failure reasons."""
    reasons = []

    # Check 1: valid JSON (already parsed if we're here)
    required_fields = [
        "source_id", "official_split", "question", "source_documents",
        "source_response", "response", "target_output", "error_type",
        "canonical_answer", "canonical_program", "corruption_method",
        "data_origin", "is_clean"
    ]
    missing = [f for f in required_fields if f not in record]
    if missing:
        reasons.append(f"missing_fields:{','.join(missing)}")
        return reasons  # Can't proceed

    is_clean = record["is_clean"]
    error_type = record["error_type"]
    response = record["response"]
    target = record["target_output"]
    canonical_answer = record.get("canonical_answer", "")

    if is_clean:
        # ── Check 12: clean examples contain no XML tags ──
        if "<numerical>" in target.lower() or "<temporal>" in target.lower():
            reasons.append("clean_has_xml_tags")
        if "<delete>" in target.lower() or "<mark>" in target.lower():
            reasons.append("clean_has_delete_mark_tags")

        # Clean target should be "Correction: [response]"
        expected = f"Correction: {response}"
        if target != expected:
            # Allow minor trailing whitespace
            if target.strip() != expected.strip():
                reasons.append("clean_target_mismatch")

        return reasons

    # ── Corrupted example checks ──

    # Check 2: exactly one error type
    if error_type not in ("numerical", "temporal"):
        reasons.append(f"invalid_error_type:{error_type}")

    # Check 3: exactly one outer XML tag
    outer_tags = XML_TAG_RE.findall(target)
    if len(outer_tags) != 1:
        reasons.append(f"outer_tags_count:{len(outer_tags)}")

    # Check 4: exactly one delete block
    deletes = DELETE_RE.findall(target)
    if len(deletes) != 1:
        reasons.append(f"delete_blocks_count:{len(deletes)}")

    # Check 5: exactly one mark block
    marks = MARK_RE.findall(target)
    if len(marks) != 1:
        reasons.append(f"mark_blocks_count:{len(marks)}")

    if len(deletes) != 1 or len(marks) != 1 or len(outer_tags) != 1:
        return reasons  # Can't proceed with further checks

    del_text = deletes[0]
    mark_text = marks[0]
    tag_name = outer_tags[0].lower()

    # Check 6: delete span appears in the corrupted response
    if del_text not in response:
        reasons.append("delete_not_in_response")

    # Check 7: mark span is the canonical correct value
    #   For Gemini rows: verified by the canonical audit
    #   For deterministic: mark should equal the original answer span
    if record["data_origin"] == "deterministic":
        source_response = record.get("source_response", "")
        if mark_text not in source_response:
            reasons.append("mark_not_in_source_response")

    # Check 8: delete and mark differ
    if del_text == mark_text:
        reasons.append("delete_equals_mark")

    # Check 9: untagged target equals the corrected response
    #   After stripping "Correction: " and replacing XML block with mark_text,
    #   we should get the source_response
    correction_prefix = "Correction: "
    if target.startswith(correction_prefix):
        corr_text = target[len(correction_prefix):]
    elif "Correction:" in target:
        corr_text = target.split("Correction:", 1)[1].strip()
    else:
        corr_text = target

    # Replace the outer XML block with mark_text
    outer_match = OUTER_BLOCK.search(corr_text)
    if outer_match:
        reconstructed_with_mark = (
            corr_text[:outer_match.start()] + mark_text + corr_text[outer_match.end():]
        )
        reconstructed_with_del = (
            corr_text[:outer_match.start()] + del_text + corr_text[outer_match.end():]
        )

        # Check 10: exactly one textual replacement
        #   reconstructed_with_del should equal the corrupted response
        if reconstructed_with_del.strip() != response.strip():
            # Allow trailing period differences
            if reconstructed_with_del.strip().rstrip(".") != response.strip().rstrip("."):
                reasons.append("reconstruction_mismatch")

        # For deterministic: reconstructed_with_mark should equal source_response
        if record["data_origin"] == "deterministic":
            source_response = record.get("source_response", "")
            if reconstructed_with_mark.strip() != source_response.strip():
                if reconstructed_with_mark.strip().rstrip(".") != source_response.strip().rstrip("."):
                    reasons.append("corrected_doesnt_match_source")

        # Check 11: no residual wrong span at the edited location
        #   After replacing XML block with mark_text, del_text should not appear
        #   at or near the edit location
        if del_text in mark_text:
            pass  # del_text is a substring of mark_text; can't avoid
        elif del_text in reconstructed_with_mark:
            # Check if del_text still appears ONLY elsewhere (not at the edit location)
            # The edit location is at outer_match.start()
            edit_start = outer_match.start()
            edit_end = edit_start + len(mark_text)
            # Check in the vicinity of the edit
            search_region = reconstructed_with_mark[max(0, edit_start-5):edit_end+5]
            if del_text in search_region:
                reasons.append("residual_wrong_span_at_edit")
    else:
        reasons.append("no_outer_xml_block_in_correction")

    # Check error_type matches tag_name
    if tag_name != error_type:
        reasons.append(f"error_type_tag_mismatch:{error_type}!={tag_name}")

    return reasons


def main():
    print("=" * 72)
    print("HYBRID DATASET VALIDATOR")
    print("=" * 72)

    all_records = {}  # split -> list of records
    all_source_ids = defaultdict(set)  # source_id -> set of splits

    total_valid = 0
    total_invalid = 0
    all_failures = []
    failure_reasons_all = Counter()
    split_stats = {}

    for split_name in SPLITS:
        path = FINAL_DIR / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue

        records = []
        valid = 0
        invalid = 0
        failure_reasons = Counter()
        failures = []

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    failures.append({
                        "split": split_name, "line": line_num,
                        "reasons": [f"invalid_json:{e}"]
                    })
                    invalid += 1
                    failure_reasons["invalid_json"] += 1
                    continue

                records.append(record)
                reasons = validate_record(record, line_num, split_name)

                if reasons:
                    invalid += 1
                    for r in reasons:
                        failure_reasons[r.split(":")[0]] += 1
                        failure_reasons_all[r.split(":")[0]] += 1
                    failures.append({
                        "split": split_name, "line": line_num,
                        "source_id": record.get("source_id", ""),
                        "reasons": reasons,
                    })
                else:
                    valid += 1

                # Track source IDs for leakage check
                sid = record.get("source_id", "")
                all_source_ids[sid].add(split_name)

        all_records[split_name] = records
        total_valid += valid
        total_invalid += invalid
        all_failures.extend(failures)

        # Split-level stats
        corrupted = [r for r in records if not r.get("is_clean", False)]
        clean = [r for r in records if r.get("is_clean", False)]
        gemini = [r for r in records if r.get("data_origin") == "gemini"]
        det = [r for r in records if r.get("data_origin") == "deterministic"]
        numerical = [r for r in records if r.get("error_type") == "numerical"]
        temporal = [r for r in records if r.get("error_type") == "temporal"]

        split_stats[split_name] = {
            "total": len(records),
            "valid": valid,
            "invalid": invalid,
            "corrupted": len(corrupted),
            "clean": len(clean),
            "gemini": len(gemini),
            "deterministic": len(det),
            "numerical": len(numerical),
            "temporal": len(temporal),
            "failure_reasons": dict(failure_reasons.most_common()),
        }

        print(f"\n  [{split_name}] total={len(records)}, valid={valid}, invalid={invalid}")
        if failure_reasons:
            for r, c in failure_reasons.most_common(5):
                print(f"    {r}: {c}")

    # ── Check 13: no source leakage across splits ──
    leaked = {sid: splits for sid, splits in all_source_ids.items() if len(splits) > 1}
    print(f"\n  Source leakage: {len(leaked)} sources in multiple splits")

    # ── Duplicate source_id check ──
    all_ids = []
    for split_name in SPLITS:
        for r in all_records.get(split_name, []):
            all_ids.append(r.get("source_id", ""))
    id_counts = Counter(all_ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}
    print(f"  Duplicate source IDs: {len(duplicates)}")

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"VALIDATION SUMMARY")
    print(f"  Total records: {total_valid + total_invalid}")
    print(f"  Valid: {total_valid}")
    print(f"  Invalid: {total_invalid}")
    print(f"  Leakage: {len(leaked)}")
    print(f"  Duplicates: {len(duplicates)}")
    if total_invalid == 0 and len(leaked) == 0:
        print(f"  STATUS: PASS")
    else:
        print(f"  STATUS: {'PASS WITH WARNINGS' if total_invalid < 50 else 'FAIL'}")
    print(f"{'='*72}")

    # ── Append to report ──
    with open(REPORT_OUT, "a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write("## 10. Post-Build Validation Results\n\n")
        f.write(f"**Validation run**: {__import__('datetime').datetime.now().isoformat()}\n\n")

        f.write("### 10.1 Per-Split Validation\n\n")
        f.write("| Split | Total | Valid | Invalid | Corrupted | Clean | Gemini | Det. | Numerical | Temporal |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for split_name in SPLITS:
            s = split_stats.get(split_name, {})
            f.write(f"| {split_name} | {s.get('total',0)} | {s.get('valid',0)} | "
                    f"{s.get('invalid',0)} | {s.get('corrupted',0)} | {s.get('clean',0)} | "
                    f"{s.get('gemini',0)} | {s.get('deterministic',0)} | "
                    f"{s.get('numerical',0)} | {s.get('temporal',0)} |\n")
        f.write("\n")

        f.write("### 10.2 Failure Reasons\n\n")
        if failure_reasons_all:
            f.write("| Reason | Count |\n")
            f.write("|---|---|\n")
            for reason, count in failure_reasons_all.most_common():
                f.write(f"| `{reason}` | {count} |\n")
        else:
            f.write("No failures detected.\n")
        f.write("\n")

        f.write("### 10.3 Source Leakage\n\n")
        if leaked:
            f.write(f"**{len(leaked)} sources in multiple splits:**\n\n")
            for sid, splits in list(leaked.items())[:10]:
                f.write(f"- `{sid}`: {splits}\n")
        else:
            f.write("No source leakage.\n")
        f.write("\n")

        f.write("### 10.4 Duplicate Source IDs\n\n")
        if duplicates:
            f.write(f"**{len(duplicates)} duplicate source IDs:**\n\n")
            for sid, count in list(sorted(duplicates.items(), key=lambda x: -x[1]))[:10]:
                f.write(f"- `{sid}`: {count}\n")
        else:
            f.write("No duplicates.\n")
        f.write("\n")

        f.write("### 10.5 Sample Failures (up to 20)\n\n")
        for i, fail in enumerate(all_failures[:20], 1):
            f.write(f"**{i}.** split=`{fail['split']}`, "
                    f"source_id=`{fail.get('source_id', 'N/A')}`, "
                    f"reasons: `{fail['reasons']}`\n\n")

        status = "PASS" if (total_invalid == 0 and len(leaked) == 0) else (
            "PASS WITH WARNINGS" if total_invalid < 50 else "FAIL"
        )
        f.write(f"### 10.6 Overall Status: **{status}**\n\n")
        f.write(f"- Total valid: {total_valid}\n")
        f.write(f"- Total invalid: {total_invalid}\n")
        f.write(f"- Source leakage: {len(leaked)}\n")
        f.write(f"- Duplicate IDs: {len(duplicates)}\n")


if __name__ == "__main__":
    main()
