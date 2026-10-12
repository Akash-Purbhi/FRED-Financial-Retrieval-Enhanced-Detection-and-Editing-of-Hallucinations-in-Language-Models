#!/usr/bin/env python3
"""
09_audit_high_confidence_canonical.py
=====================================
Canonical audit of the high-confidence Gemini-generated CoT examples
against the official FinQA ground truth.

Input:  synthetic_finqa_cot_5000_grounded.jsonl  (1,324 rows)
        ─ re-derives the 1,155 strict-valid subset using the same logic
          as validate_cot_dataset.py, then runs the 5-point canonical audit.

Refs:   data/official_finqa/{train,dev,test}.json

Output: data/audit/high_confidence_canonical_valid.jsonl
        data/audit/high_confidence_canonical_rejected.jsonl
        reports/high_confidence_canonical_report.md

Checks per row:
  1. Canonical answer agreement  – <mark> value matches official FinQA answer
  2. Exactly one textual replacement – single <delete>/<mark> pair
  3. Correct delete/mark direction – <delete> ≠ <mark>, <delete> is the WRONG
     value, <mark> is the CORRECT value
  4. No repeated wrong span remains – after the correction the wrong value
     should NOT appear elsewhere in the corrected passage
  5. Numerical or temporal taxonomy is valid – outer tag is <numerical> or
     <temporal> and the content matches the declared type
"""

import json
import re
import sys
import difflib
from pathlib import Path
from collections import Counter, defaultdict

# ── paths ────────────────────────────────────────────────────────────────────
WORKSPACE   = Path(__file__).resolve().parent.parent
GROUNDED    = WORKSPACE / "synthetic_finqa_cot_5000_grounded.jsonl"
OFFICIAL    = WORKSPACE / "data" / "official_finqa"
AUDIT_DIR   = WORKSPACE / "data" / "audit"
REPORT_DIR  = WORKSPACE / "reports"
VALID_OUT   = AUDIT_DIR / "high_confidence_canonical_valid.jsonl"
REJECTED_OUT= AUDIT_DIR / "high_confidence_canonical_rejected.jsonl"
REPORT_OUT  = REPORT_DIR / "high_confidence_canonical_report.md"

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def norm(text: str) -> str:
    """Lowercase, strip non-alphanumeric for fuzzy key matching."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def norm_number(val: str) -> str:
    """Normalise a number string: strip $, %, commas, trailing zeros."""
    v = val.strip().replace(",", "").replace("$", "").rstrip("%")
    try:
        f = float(v)
        # Return a canonical float representation
        return f"{f:g}"
    except ValueError:
        return v.lower().strip()


def load_official_finqa() -> dict:
    """Load all three official FinQA splits into a dict keyed by normalised
    question text.  Each value is a list of official records (to handle the
    rare case of duplicate questions across splits)."""
    index: dict[str, list] = defaultdict(list)
    split_counts = {}
    for split in ("train", "dev", "test"):
        path = OFFICIAL / f"{split}.json"
        if not path.exists():
            print(f"  [WARN] {path} not found — skipping")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        split_counts[split] = len(data)
        for item in data:
            q = norm(item["qa"]["question"])
            item["_split"] = split  # annotate for reporting
            index[q].append(item)
    print(f"  Loaded official FinQA — {split_counts}")
    return dict(index)


# ── stage 1: re-derive XML-strict valid subset (same as validate_cot_dataset) ─
def xml_validate(row: dict) -> list[str]:
    """Return a list of rejection reasons.  Empty list → valid."""
    reasons = []
    target = row.get("target_output", "")
    response = row.get("response", "")

    # Correction section
    if target.count("Correction:") != 1:
        reasons.append("not_exactly_one_correction_section")

    # Outer tag
    outer_tags = re.findall(r"<(numerical|temporal)>", target, re.I)
    if len(outer_tags) != 1:
        reasons.append("not_exactly_one_outer_tag")

    # Delete / Mark blocks
    deletes = re.findall(r"<delete>(.*?)</delete>", target, re.DOTALL)
    marks   = re.findall(r"<mark>(.*?)</mark>",     target, re.DOTALL)
    if len(deletes) != 1:
        reasons.append("not_exactly_one_delete_block")
    if len(marks) != 1:
        reasons.append("not_exactly_one_mark_block")

    # Tag closure
    for tag in ("numerical", "temporal", "delete", "mark"):
        opens  = len(re.findall(f"<{tag}>",  target, re.I))
        closes = len(re.findall(f"</{tag}>", target, re.I))
        if opens != closes:
            reasons.append(f"unclosed_tag_{tag}")

    if len(deletes) == 1 and len(marks) == 1 and len(outer_tags) == 1:
        del_text  = deletes[0]
        mark_text = marks[0]

        # Wrong span must appear in original erroneous response
        if del_text not in response:
            reasons.append("wrong_span_not_in_response")

        # Delete and mark must differ
        if del_text == mark_text:
            reasons.append("delete_and_mark_identical")

        # Reconstruction: correction text with tag block replaced by del_text
        # should equal the erroneous response
        idx = target.find("Correction:")
        if idx != -1:
            corr_text = target[idx + len("Correction:"):].strip()
            tag_name = outer_tags[0]
            tag_block_pat = re.compile(
                rf"<{tag_name}>.*?</{tag_name}>", re.I | re.DOTALL
            )
            m = tag_block_pat.search(corr_text)
            if m:
                reconstructed = corr_text.replace(m.group(0), del_text)
                if reconstructed.strip().rstrip(".") != response.strip().rstrip("."):
                    reasons.append("unrelated_text_changed_or_bad_replacement")
            else:
                reasons.append("malformed_outer_tag_block")

    return reasons


# ── stage 2: 5-point canonical audit ─────────────────────────────────────────

def canonical_audit(row: dict, official_index: dict) -> dict:
    """Run the 5-point audit.  Returns a dict with audit results."""
    result = {
        "pass": True,
        "reasons": [],
        "matched_split": None,
        "canonical_answer": None,
        "canonical_exe_ans": None,
    }

    target   = row.get("target_output", "")
    response = row.get("response", "")

    # Parse edit components
    outer_tags = re.findall(r"<(numerical|temporal)>", target, re.I)
    deletes    = re.findall(r"<delete>(.*?)</delete>", target, re.DOTALL)
    marks      = re.findall(r"<mark>(.*?)</mark>",     target, re.DOTALL)

    if not (len(outer_tags) == 1 and len(deletes) == 1 and len(marks) == 1):
        result["pass"] = False
        result["reasons"].append("structural_fail_skipped")
        return result

    tag_name  = outer_tags[0]
    del_text  = deletes[0]
    mark_text = marks[0]

    # ── match to official FinQA ──
    q_norm = norm(row.get("question", ""))
    officials = official_index.get(q_norm, [])

    if not officials:
        result["pass"] = False
        result["reasons"].append("no_finqa_match")
        return result

    if len(officials) > 1:
        # Try to disambiguate by checking which official answer matches mark_text
        filtered = [
            o for o in officials
            if norm_number(str(o["qa"].get("exe_ans", ""))) == norm_number(mark_text)
            or norm_number(str(o["qa"].get("answer", ""))) == norm_number(mark_text)
        ]
        if len(filtered) == 1:
            officials = filtered
        # else: keep all and proceed — may be flagged

    official = officials[0]
    result["matched_split"] = official.get("_split", "unknown")
    exe_ans  = str(official["qa"].get("exe_ans", ""))
    text_ans = str(official["qa"].get("answer", ""))
    result["canonical_answer"]  = text_ans
    result["canonical_exe_ans"] = exe_ans

    # ── CHECK 1: canonical answer agreement ──
    mark_norm = norm_number(mark_text)
    ans_variants = {
        norm_number(exe_ans),
        norm_number(text_ans),
    }
    # Also try percentage equivalence: 0.25 ↔ 25%  or 25 ↔ 25%
    try:
        exe_f = float(exe_ans)
        ans_variants.add(norm_number(str(exe_f * 100)))
        ans_variants.add(norm_number(str(exe_f)))
    except (ValueError, TypeError):
        pass

    # Flexible containment check (some answers are embedded in longer text)
    canonical_match = (
        mark_norm in ans_variants
        or any(mark_norm in av for av in ans_variants)
        or any(av in mark_norm for av in ans_variants if av)
    )
    if not canonical_match:
        result["pass"] = False
        result["reasons"].append("canonical_answer_mismatch")

    # ── CHECK 2: exactly one textual replacement ──
    #   (already guaranteed by xml_validate, but double-check via diff)
    idx = target.find("Correction:")
    if idx != -1:
        corr_text = target[idx + len("Correction:"):].strip()
        tag_block_pat = re.compile(
            rf"<{tag_name}>.*?</{tag_name}>", re.I | re.DOTALL
        )
        m = tag_block_pat.search(corr_text)
        if m:
            # text_with_correction = corr_text with tag block replaced by mark_text
            corrected_full = corr_text.replace(m.group(0), mark_text)
            erroneous_full = corr_text.replace(m.group(0), del_text)

            # Count diff blocks between erroneous_full and response
            sm = difflib.SequenceMatcher(None, response.strip(), erroneous_full.strip())
            diff_ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
            if len(diff_ops) > 0:
                # More than trivial whitespace differences
                # Allow minor trailing-period differences
                stripped_r = response.strip().rstrip(".")
                stripped_e = erroneous_full.strip().rstrip(".")
                if stripped_r != stripped_e:
                    result["reasons"].append("multi_edit_or_paraphrase")
                    result["pass"] = False

    # ── CHECK 3: correct delete/mark direction ──
    #   <delete> should be the wrong value (hallucinated, not in context)
    #   <mark>   should be the correct value (the canonical answer)
    # We already checked mark vs canonical above.
    # Additionally verify delete ≠ canonical answer
    del_norm = norm_number(del_text)
    if del_norm in ans_variants:
        result["pass"] = False
        result["reasons"].append("delete_is_canonical_answer_reversed")

    # ── CHECK 4: no repeated wrong span remains ──
    #   After applying the correction, the wrong value should not appear
    #   in the corrected passage.
    if idx != -1 and m:
        corrected_full = corr_text.replace(m.group(0), mark_text)
        # Check if del_text still appears in corrected_full
        # But exclude cases where del_text is a substring of mark_text
        if del_text in corrected_full and del_text not in mark_text:
            result["pass"] = False
            result["reasons"].append("wrong_span_remains_after_correction")

    # ── CHECK 5: numerical or temporal taxonomy validity ──
    #   Check that the tag type matches the content
    is_temporal = bool(re.search(
        r"(year|month|date|quarter|period|week|day|jan|feb|mar|apr|may|jun|"
        r"jul|aug|sep|oct|nov|dec|q[1-4]|fy\s?\d|20[012]\d|19\d\d|"
        r"first\s+half|second\s+half|fiscal|annual|semi.?annual|"
        r"beginning|ending|ended)",
        del_text + " " + mark_text,
        re.I
    ))
    is_numerical = bool(re.search(
        r"(\d+\.?\d*\s*%?|\$\s*[\d,.]+|million|billion|thousand|ratio|"
        r"rate|percent|growth|decline|increase|decrease|average|total|sum|"
        r"difference|margin|revenue|income|loss|expense|cost|price|"
        r"earnings|dividend|eps|roi|ebitda|wacc)",
        del_text + " " + mark_text,
        re.I
    ))

    if tag_name.lower() == "temporal" and not is_temporal:
        result["reasons"].append("temporal_tag_but_no_temporal_content")
        # Soft fail — don't reject if otherwise valid
    elif tag_name.lower() == "numerical" and not is_numerical:
        result["reasons"].append("numerical_tag_but_no_numerical_content")
        # Soft fail
    elif tag_name.lower() == "temporal" and is_numerical and not is_temporal:
        result["reasons"].append("should_be_numerical_not_temporal")
        result["pass"] = False
    elif tag_name.lower() not in ("numerical", "temporal"):
        result["reasons"].append("unknown_taxonomy_tag")
        result["pass"] = False

    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("HIGH-CONFIDENCE CANONICAL AUDIT")
    print("=" * 72)

    # Step 0: load official FinQA
    print("\n[1/4] Loading official FinQA ground truth …")
    official_index = load_official_finqa()
    print(f"  → {len(official_index)} unique normalised questions indexed")

    # Step 1: load grounded file
    print("\n[2/4] Loading grounded rows …")
    if not GROUNDED.exists():
        print(f"  ERROR: {GROUNDED} not found"); sys.exit(1)
    grounded = []
    with open(GROUNDED, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                grounded.append(json.loads(line))
    print(f"  → {len(grounded)} grounded rows loaded")

    # Step 2: re-derive XML-valid subset (same as validate_cot_dataset.py)
    print("\n[3/4] XML structural validation (re-deriving high-confidence set) …")
    seen_ids = set()
    xml_valid = []
    xml_rejected = []
    xml_rejection_stats = Counter()

    for row in grounded:
        sid = row.get("source_id", "")
        if sid in seen_ids:
            xml_rejected.append(row)
            xml_rejection_stats["duplicate_source_id"] += 1
            continue
        if sid:
            seen_ids.add(sid)

        reasons = xml_validate(row)
        if reasons:
            row["xml_rejection_reasons"] = reasons
            xml_rejected.append(row)
            for r in reasons:
                xml_rejection_stats[r] += 1
        else:
            xml_valid.append(row)

    print(f"  → XML-valid: {len(xml_valid)},  XML-rejected: {len(xml_rejected)}")

    # Step 3: canonical 5-point audit on the XML-valid rows
    print("\n[4/4] Running 5-point canonical audit …")
    canonical_valid   = []
    canonical_rejected = []
    audit_stats = Counter()
    taxonomy_dist = Counter()
    split_dist = Counter()
    rejection_detail = Counter()
    check1_examples = []  # canonical mismatch examples for report
    check3_examples = []  # reversed direction examples
    check4_examples = []  # remaining wrong span examples

    for row in xml_valid:
        audit = canonical_audit(row, official_index)

        # Add audit metadata to the row (non-destructive copy)
        row_out = dict(row)
        row_out["audit"] = audit

        if audit["pass"]:
            canonical_valid.append(row_out)
            audit_stats["canonical_pass"] += 1

            # Taxonomy distribution
            tags = re.findall(r"<(numerical|temporal)>", row.get("target_output", ""), re.I)
            if tags:
                taxonomy_dist[tags[0].lower()] += 1

            if audit["matched_split"]:
                split_dist[audit["matched_split"]] += 1
        else:
            canonical_rejected.append(row_out)
            audit_stats["canonical_fail"] += 1
            for reason in audit["reasons"]:
                rejection_detail[reason] += 1

            # Collect examples
            if "canonical_answer_mismatch" in audit["reasons"] and len(check1_examples) < 5:
                check1_examples.append(row_out)
            if "delete_is_canonical_answer_reversed" in audit["reasons"] and len(check3_examples) < 5:
                check3_examples.append(row_out)
            if "wrong_span_remains_after_correction" in audit["reasons"] and len(check4_examples) < 5:
                check4_examples.append(row_out)

        # Track soft warnings
        for reason in audit["reasons"]:
            if reason.endswith("_content"):
                audit_stats[f"warning_{reason}"] += 1

    # Also count unmatched rows
    no_match_count = rejection_detail.get("no_finqa_match", 0)

    print(f"  → Canonical PASS: {audit_stats['canonical_pass']}")
    print(f"  → Canonical FAIL: {audit_stats['canonical_fail']}")

    # ── write outputs ────────────────────────────────────────────────────
    print("\nWriting outputs …")
    with open(VALID_OUT, "w", encoding="utf-8") as f:
        for r in canonical_valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → {VALID_OUT}  ({len(canonical_valid)} rows)")

    with open(REJECTED_OUT, "w", encoding="utf-8") as f:
        for r in canonical_rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Also write the XML-rejected rows
        for r in xml_rejected:
            r_out = dict(r)
            r_out["audit"] = {"pass": False, "reasons": r.get("xml_rejection_reasons", ["xml_fail"])}
            f.write(json.dumps(r_out, ensure_ascii=False) + "\n")
    total_rejected = len(canonical_rejected) + len(xml_rejected)
    print(f"  → {REJECTED_OUT}  ({total_rejected} rows)")

    # ── write report ─────────────────────────────────────────────────────
    with open(REPORT_OUT, "w", encoding="utf-8") as rpt:
        rpt.write("# High-Confidence Canonical Audit Report\n\n")
        rpt.write(f"**Date**: {__import__('datetime').datetime.now().isoformat()}\n\n")

        rpt.write("## 1. Pipeline Summary\n\n")
        rpt.write("| Stage | Count |\n")
        rpt.write("|---|---|\n")
        rpt.write(f"| Raw 5,000 CoT rows | 5,000 |\n")
        rpt.write(f"| Grounded (fuzzy-matched to RagBench) | {len(grounded)} |\n")
        rpt.write(f"| XML-structurally valid | {len(xml_valid)} |\n")
        rpt.write(f"| XML-rejected | {len(xml_rejected)} |\n")
        rpt.write(f"| Canonical audit PASS | {audit_stats['canonical_pass']} |\n")
        rpt.write(f"| Canonical audit FAIL | {audit_stats['canonical_fail']} |\n")
        rpt.write(f"| **Final valid rows** | **{len(canonical_valid)}** |\n")
        rpt.write(f"| **Total rejected rows** | **{total_rejected}** |\n\n")

        rpt.write("## 2. XML Structural Rejection Reasons\n\n")
        rpt.write("| Reason | Count |\n")
        rpt.write("|---|---|\n")
        for reason, count in xml_rejection_stats.most_common():
            rpt.write(f"| {reason} | {count} |\n")
        rpt.write("\n")

        rpt.write("## 3. Canonical Audit — 5-Point Check Results\n\n")
        rpt.write("### 3.1 Rejection Reason Breakdown\n\n")
        rpt.write("| Reason | Count | Description |\n")
        rpt.write("|---|---|---|\n")
        reason_descriptions = {
            "no_finqa_match": "Question not found in any official FinQA split",
            "canonical_answer_mismatch": "`<mark>` value ≠ official FinQA answer",
            "multi_edit_or_paraphrase": "Correction text ≠ single substitution of response",
            "delete_is_canonical_answer_reversed": "`<delete>` tag contains the *correct* answer (direction reversed)",
            "wrong_span_remains_after_correction": "Wrong value still present after applying correction",
            "should_be_numerical_not_temporal": "Tagged `<temporal>` but content is numerical",
            "unknown_taxonomy_tag": "Outer tag is not `<numerical>` or `<temporal>`",
            "structural_fail_skipped": "Structural pre-check failed (shouldn't reach here)",
        }
        for reason, count in rejection_detail.most_common():
            desc = reason_descriptions.get(reason, "—")
            rpt.write(f"| `{reason}` | {count} | {desc} |\n")
        rpt.write("\n")

        rpt.write("### 3.2 Taxonomy Distribution (Valid Rows)\n\n")
        rpt.write("| Type | Count | % |\n")
        rpt.write("|---|---|---|\n")
        total_valid = len(canonical_valid) or 1
        for tag, count in taxonomy_dist.most_common():
            rpt.write(f"| `<{tag}>` | {count} | {count/total_valid*100:.1f}% |\n")
        rpt.write("\n")

        rpt.write("### 3.3 FinQA Split Distribution (Valid Rows)\n\n")
        rpt.write("| Split | Count |\n")
        rpt.write("|---|---|\n")
        for split, count in split_dist.most_common():
            rpt.write(f"| {split} | {count} |\n")
        rpt.write("\n")

        # Soft warnings
        warnings = {k: v for k, v in audit_stats.items() if k.startswith("warning_")}
        if warnings:
            rpt.write("### 3.4 Taxonomy Soft Warnings (not rejected)\n\n")
            rpt.write("| Warning | Count |\n")
            rpt.write("|---|---|\n")
            for w, c in sorted(warnings.items(), key=lambda x: -x[1]):
                rpt.write(f"| `{w}` | {c} |\n")
            rpt.write("\n")

        # ── Examples ──
        rpt.write("## 4. Example Failures\n\n")

        if check1_examples:
            rpt.write("### 4.1 Canonical Answer Mismatch (Check 1)\n\n")
            for i, ex in enumerate(check1_examples, 1):
                a = ex["audit"]
                rpt.write(f"**Example {i}** — source_id: `{ex.get('source_id', 'N/A')}`\n\n")
                rpt.write(f"- Question: {ex.get('question', '')}\n")
                deletes = re.findall(r"<delete>(.*?)</delete>", ex.get("target_output", ""), re.DOTALL)
                marks   = re.findall(r"<mark>(.*?)</mark>",     ex.get("target_output", ""), re.DOTALL)
                rpt.write(f"- `<delete>`: `{deletes[0] if deletes else 'N/A'}`\n")
                rpt.write(f"- `<mark>`: `{marks[0] if marks else 'N/A'}`\n")
                rpt.write(f"- Official answer: `{a.get('canonical_answer', '')}` / exe_ans: `{a.get('canonical_exe_ans', '')}`\n\n")

        if check3_examples:
            rpt.write("### 4.2 Reversed Direction (Check 3)\n\n")
            for i, ex in enumerate(check3_examples, 1):
                a = ex["audit"]
                rpt.write(f"**Example {i}** — source_id: `{ex.get('source_id', 'N/A')}`\n\n")
                rpt.write(f"- Question: {ex.get('question', '')}\n")
                deletes = re.findall(r"<delete>(.*?)</delete>", ex.get("target_output", ""), re.DOTALL)
                marks   = re.findall(r"<mark>(.*?)</mark>",     ex.get("target_output", ""), re.DOTALL)
                rpt.write(f"- `<delete>` (should be wrong): `{deletes[0] if deletes else 'N/A'}`\n")
                rpt.write(f"- `<mark>` (should be correct): `{marks[0] if marks else 'N/A'}`\n")
                rpt.write(f"- Official answer: `{a.get('canonical_answer', '')}` / exe_ans: `{a.get('canonical_exe_ans', '')}`\n\n")

        if check4_examples:
            rpt.write("### 4.3 Wrong Span Remains After Correction (Check 4)\n\n")
            for i, ex in enumerate(check4_examples, 1):
                rpt.write(f"**Example {i}** — source_id: `{ex.get('source_id', 'N/A')}`\n\n")
                rpt.write(f"- Question: {ex.get('question', '')}\n")
                deletes = re.findall(r"<delete>(.*?)</delete>", ex.get("target_output", ""), re.DOTALL)
                rpt.write(f"- Wrong span: `{deletes[0] if deletes else 'N/A'}`\n\n")

        # ── Verdict ──
        rpt.write("## 5. Verdict: Are the Existing Rows Sufficient?\n\n")

        enough_corrupted = len(canonical_valid) >= 800
        target_min_corrupted = 5000
        target_max_corrupted = 6000
        target_min_clean = 1000
        target_max_clean = 1500

        rpt.write(f"The hybrid dataset plan requires:\n")
        rpt.write(f"- **{target_min_corrupted}–{target_max_corrupted}** corrupted examples\n")
        rpt.write(f"- **{target_min_clean}–{target_max_clean}** clean examples\n\n")

        rpt.write(f"### Canonical-valid Gemini examples available: **{len(canonical_valid)}**\n\n")

        if len(canonical_valid) >= target_min_corrupted:
            rpt.write("> ✅ The existing Gemini examples alone meet the corrupted target.\n\n")
        else:
            deficit = target_min_corrupted - len(canonical_valid)
            rpt.write(f"> ⚠️ **Shortfall**: {len(canonical_valid)} canonical-valid Gemini examples ")
            rpt.write(f"fall **{deficit}** short of the minimum {target_min_corrupted} corrupted target.\n\n")
            rpt.write(f"To reach {target_min_corrupted}–{target_max_corrupted} corrupted examples, ")
            rpt.write(f"**{deficit}–{target_max_corrupted - len(canonical_valid)}** additional examples ")
            rpt.write(f"must be generated via deterministic Python-based numerical/temporal corruption.\n\n")

        rpt.write("### Clean examples: **0** (must be generated separately)\n\n")
        rpt.write(f"All {len(canonical_valid)} valid rows are corrupted (contain errors).\n")
        rpt.write(f"**{target_min_clean}–{target_max_clean}** clean pass-through examples must be ")
        rpt.write(f"constructed from official FinQA rows where the model response already ")
        rpt.write(f"matches the canonical answer.\n\n")

        rpt.write("### Recommendation\n\n")
        if len(canonical_valid) < target_min_corrupted:
            rpt.write(f"1. **Retain** the {len(canonical_valid)} canonical-valid Gemini examples as natural-language corrupted training data.\n")
            rpt.write(f"2. **Supplement** with {deficit}–{target_max_corrupted - len(canonical_valid)} deterministic Python-generated numerical/temporal errors.\n")
            rpt.write(f"3. **Generate** {target_min_clean}–{target_max_clean} clean pass-through examples from official FinQA.\n")
            rpt.write(f"4. **Remove** free-form Chain-of-Thought from the primary training format.\n")
            rpt.write(f"5. **Enforce** exactly one XML edit per target.\n")
        else:
            rpt.write(f"1. **Use** the {len(canonical_valid)} canonical-valid Gemini examples directly.\n")
            rpt.write(f"2. **Generate** {target_min_clean}–{target_max_clean} clean pass-through examples.\n")

    print(f"\n  → Report: {REPORT_OUT}")
    print("\n" + "=" * 72)
    print("AUDIT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
