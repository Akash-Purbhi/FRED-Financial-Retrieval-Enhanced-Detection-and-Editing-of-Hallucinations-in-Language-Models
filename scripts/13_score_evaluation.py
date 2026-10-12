"""
FRED Evaluation — Step 2: Score Predictions
=============================================
Parses predictions from both models, computes paper-aligned metrics,
and generates the evaluation report.

Metrics (paper Table 3 format):
  - Precision, Recall, F1 per error type (Numerical, Temporal)
  - Overall Precision, Recall, F1
  - Binary Detection Precision, Recall, F1

Additional deterministic editing metrics:
  - XML Validity Rate
  - Wrong-Span (Delete) Accuracy
  - Corrected-Span (Mark) Accuracy
  - Complete Edit Success (CES)
  - Clean-Response Preservation Rate

Output:
  - data/eval/scores.json
  - reports/evaluation_report.md
"""

import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Config ───────────────────────────────────────────────────────────────────
EVAL_DIR    = Path("data/eval")
REPORT_PATH = Path("reports/evaluation_report.md")
SCORES_PATH = EVAL_DIR / "scores.json"

PRED_FILES = {
    "fred":  EVAL_DIR / "predictions_fred.jsonl",
    "base":  EVAL_DIR / "predictions_base.jsonl",
}

# ── XML Parsing ──────────────────────────────────────────────────────────────

OUTER_RE   = re.compile(r"<(numerical|temporal)>(.*?)</\1>", re.I | re.DOTALL)
DELETE_RE  = re.compile(r"<delete>(.*?)</delete>", re.I | re.DOTALL)
MARK_RE    = re.compile(r"<mark>(.*?)</mark>", re.I | re.DOTALL)
ANY_TAG_RE = re.compile(r"</?(?:numerical|temporal|delete|mark)>", re.I)


def parse_prediction(text):
    """Parse a model prediction into structured fields."""
    result = {
        "has_edit": False,
        "edit_type": None,
        "delete_span": None,
        "mark_span": None,
        "xml_valid": False,
        "num_outer_tags": 0,
        "num_delete_tags": 0,
        "num_mark_tags": 0,
        "raw_text": text,
    }

    if not text:
        return result

    # Strip "Correction: " prefix if present
    clean = text
    if clean.lower().startswith("correction:"):
        clean = clean[len("correction:"):].strip()

    # Find outer tags
    outer_matches = list(OUTER_RE.finditer(clean))
    result["num_outer_tags"] = len(outer_matches)

    if len(outer_matches) == 0:
        # No XML tags found — model thinks it's clean
        result["has_edit"] = False
        result["xml_valid"] = True  # valid "no edit" response
        return result

    result["has_edit"] = True

    if len(outer_matches) == 1:
        m = outer_matches[0]
        result["edit_type"] = m.group(1).lower()
        inner = m.group(2)

        deletes = DELETE_RE.findall(inner)
        marks = MARK_RE.findall(inner)

        result["num_delete_tags"] = len(deletes)
        result["num_mark_tags"] = len(marks)

        if len(deletes) == 1 and len(marks) == 1:
            result["delete_span"] = deletes[0]
            result["mark_span"] = marks[0]
            result["xml_valid"] = True
        else:
            result["xml_valid"] = False
    else:
        # Multiple outer tags — count all inner tags across all
        result["edit_type"] = outer_matches[0].group(1).lower()
        all_deletes = []
        all_marks = []
        for m in outer_matches:
            inner = m.group(2)
            all_deletes.extend(DELETE_RE.findall(inner))
            all_marks.extend(MARK_RE.findall(inner))
        result["num_delete_tags"] = len(all_deletes)
        result["num_mark_tags"] = len(all_marks)
        if len(all_deletes) >= 1:
            result["delete_span"] = all_deletes[0]
        if len(all_marks) >= 1:
            result["mark_span"] = all_marks[0]
        result["xml_valid"] = False  # multi-tag is invalid for our format

    return result


def parse_ground_truth(target_output, error_type, is_clean):
    """Parse the ground truth target into structured fields."""
    result = {
        "has_edit": not is_clean,
        "edit_type": error_type if error_type != "none" else None,
        "delete_span": None,
        "mark_span": None,
    }

    if is_clean:
        return result

    clean = target_output
    if clean.lower().startswith("correction:"):
        clean = clean[len("correction:"):].strip()

    outer_matches = list(OUTER_RE.finditer(clean))
    if outer_matches:
        m = outer_matches[0]
        result["edit_type"] = m.group(1).lower()
        inner = m.group(2)
        deletes = DELETE_RE.findall(inner)
        marks = MARK_RE.findall(inner)
        if deletes:
            result["delete_span"] = deletes[0]
        if marks:
            result["mark_span"] = marks[0]

    return result


# ── Metrics Computation ─────────────────────────────────────────────────────

def compute_f1(tp, fp, fn):
    """Compute precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision * 100, 1),
        "recall":    round(recall * 100, 1),
        "f1":        round(f1 * 100, 1),
        "tp": tp, "fp": fp, "fn": fn,
    }


def score_model(predictions):
    """Compute all metrics for a single model's predictions."""

    # ── Per-type detection counters ──────────────────────────────────────
    # For each error type: TP = model correctly detected that type
    #                      FP = model predicted that type but GT was different/clean
    #                      FN = GT was that type but model didn't detect it
    type_counters = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # ── Binary detection counters ────────────────────────────────────────
    binary = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    # ── Editing metrics ──────────────────────────────────────────────────
    xml_valid_count = 0
    xml_total_with_tags = 0
    delete_correct = 0
    mark_correct = 0
    complete_edit_success = 0
    edit_attempts = 0

    # ── Clean preservation ───────────────────────────────────────────────
    clean_total = 0
    clean_preserved = 0

    # ── Track failures for examples ──────────────────────────────────────
    failures = []
    successes = []

    for pred in predictions:
        gt = parse_ground_truth(
            pred["target_output"], pred["error_type"], pred["is_clean"]
        )
        pd_parsed = parse_prediction(pred["prediction"])

        gt_has_edit = gt["has_edit"]
        pd_has_edit = pd_parsed["has_edit"]

        # ── Binary detection ─────────────────────────────────────────────
        if gt_has_edit and pd_has_edit:
            binary["tp"] += 1
        elif gt_has_edit and not pd_has_edit:
            binary["fn"] += 1
        elif not gt_has_edit and pd_has_edit:
            binary["fp"] += 1
        else:
            binary["tn"] += 1

        # ── Per-type detection ───────────────────────────────────────────
        gt_type = gt["edit_type"]
        pd_type = pd_parsed["edit_type"]

        if gt_has_edit and gt_type:
            if pd_has_edit and pd_type == gt_type:
                type_counters[gt_type]["tp"] += 1
            else:
                type_counters[gt_type]["fn"] += 1

        if pd_has_edit and pd_type:
            if not gt_has_edit or pd_type != gt_type:
                type_counters[pd_type]["fp"] += 1

        # ── Clean preservation ───────────────────────────────────────────
        if not gt_has_edit:
            clean_total += 1
            # Check if prediction has no XML tags
            if not pd_has_edit:
                clean_preserved += 1

        # ── Editing accuracy (only when GT has an edit) ──────────────────
        if gt_has_edit and pd_has_edit:
            edit_attempts += 1

            if pd_parsed["num_outer_tags"] > 0:
                xml_total_with_tags += 1
                if pd_parsed["xml_valid"]:
                    xml_valid_count += 1

            # Delete span match
            del_match = (pd_parsed["delete_span"] is not None and
                         gt["delete_span"] is not None and
                         pd_parsed["delete_span"].strip() == gt["delete_span"].strip())
            if del_match:
                delete_correct += 1

            # Mark span match
            mark_match = (pd_parsed["mark_span"] is not None and
                          gt["mark_span"] is not None and
                          pd_parsed["mark_span"].strip() == gt["mark_span"].strip())
            if mark_match:
                mark_correct += 1

            # Complete edit success: correct type + correct delete + correct mark + valid XML
            if (pd_type == gt_type and del_match and mark_match and
                    pd_parsed["xml_valid"]):
                complete_edit_success += 1
                successes.append(pred)
            else:
                failures.append({
                    "source_id": pred["source_id"],
                    "gt_type": gt_type,
                    "pd_type": pd_type,
                    "gt_delete": gt["delete_span"],
                    "pd_delete": pd_parsed["delete_span"],
                    "gt_mark": gt["mark_span"],
                    "pd_mark": pd_parsed["mark_span"],
                    "xml_valid": pd_parsed["xml_valid"],
                })
        elif gt_has_edit and not pd_has_edit:
            failures.append({
                "source_id": pred["source_id"],
                "gt_type": gt_type,
                "pd_type": None,
                "reason": "missed_error",
            })

    # ── Compute F1 scores ────────────────────────────────────────────────
    type_f1 = {}
    for t in ["numerical", "temporal"]:
        c = type_counters[t]
        type_f1[t] = compute_f1(c["tp"], c["fp"], c["fn"])

    # Overall: macro average of all types that have support
    all_tp = sum(type_counters[t]["tp"] for t in type_counters)
    all_fp = sum(type_counters[t]["fp"] for t in type_counters)
    all_fn = sum(type_counters[t]["fn"] for t in type_counters)
    overall_f1 = compute_f1(all_tp, all_fp, all_fn)

    binary_f1 = compute_f1(binary["tp"], binary["fp"], binary["fn"])
    binary_f1["tn"] = binary["tn"]

    # ── Editing metrics ──────────────────────────────────────────────────
    editing = {
        "xml_valid_rate": round(xml_valid_count / xml_total_with_tags * 100, 1)
                          if xml_total_with_tags > 0 else 0.0,
        "xml_valid_count": xml_valid_count,
        "xml_total_with_tags": xml_total_with_tags,
        "delete_accuracy": round(delete_correct / edit_attempts * 100, 1)
                           if edit_attempts > 0 else 0.0,
        "mark_accuracy": round(mark_correct / edit_attempts * 100, 1)
                         if edit_attempts > 0 else 0.0,
        "complete_edit_success": round(complete_edit_success / edit_attempts * 100, 1)
                                 if edit_attempts > 0 else 0.0,
        "edit_attempts": edit_attempts,
        "delete_correct": delete_correct,
        "mark_correct": mark_correct,
        "ces_count": complete_edit_success,
    }

    clean_metrics = {
        "total": clean_total,
        "preserved": clean_preserved,
        "rate": round(clean_preserved / clean_total * 100, 1) if clean_total > 0 else 0.0,
    }

    return {
        "binary": binary_f1,
        "numerical": type_f1.get("numerical", compute_f1(0, 0, 0)),
        "temporal": type_f1.get("temporal", compute_f1(0, 0, 0)),
        "overall": overall_f1,
        "editing": editing,
        "clean": clean_metrics,
        "sample_failures": failures[:20],
        "sample_successes": successes[:5],
    }


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(all_scores):
    lines = []
    lines.append("# FRED Evaluation Report")
    lines.append(f"\n**Generated**: {datetime.now().isoformat()}")
    lines.append(f"\n**Test set**: 841 records (651 corrupted + 190 clean)")
    lines.append(f"\n**Models compared**: fred:latest (fine-tuned) vs qwen2.5:7b (base)")

    # ── Table 3 format: Detection F1 ─────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 1. Detection Performance (F1-score) — Paper Table 3 Format")
    lines.append("")
    lines.append("| Editor | Num. | Temp. | Overall | Binary |")
    lines.append("|---|---|---|---|---|")

    # Paper reference rows
    lines.append("| GPT-4.1 mini *(paper)* | 23.5 | 24.6 | 46.0 | 77.8 |")
    lines.append("| o3 *(paper)* | 52.2 | 83.8 | 71.9 | 90.3 |")
    lines.append("| Phi4-mini-8k* *(paper)* | 29.1 | 51.9 | 63.9 | 83.8 |")
    lines.append("| QWEN3-4b-36k* *(paper)* | 48.2 | 58.8 | 72.4 | 89.7 |")
    lines.append("| Phi4-8k* *(paper)* | 71.4 | 85.7 | 89.9 | 96.7 |")

    for model_key in ["base", "fred"]:
        if model_key not in all_scores:
            continue
        s = all_scores[model_key]
        label = "qwen2.5:7b (base)" if model_key == "base" else "**fred:latest (ours)** *"
        lines.append(f"| {label} | {s['numerical']['f1']} | {s['temporal']['f1']} "
                     f"| {s['overall']['f1']} | {s['binary']['f1']} |")

    lines.append("\n\\* fine-tuned model")

    # ── Detection Precision ──────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 2. Detection Performance (Precision) — Paper Table 11 Format")
    lines.append("")
    lines.append("| Editor | Num. | Temp. | Overall | Binary |")
    lines.append("|---|---|---|---|---|")
    for model_key in ["base", "fred"]:
        if model_key not in all_scores:
            continue
        s = all_scores[model_key]
        label = "qwen2.5:7b (base)" if model_key == "base" else "**fred:latest (ours)** *"
        lines.append(f"| {label} | {s['numerical']['precision']} | {s['temporal']['precision']} "
                     f"| {s['overall']['precision']} | {s['binary']['precision']} |")

    # ── Detection Recall ─────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 3. Detection Performance (Recall) — Paper Table 12 Format")
    lines.append("")
    lines.append("| Editor | Num. | Temp. | Overall | Binary |")
    lines.append("|---|---|---|---|---|")
    for model_key in ["base", "fred"]:
        if model_key not in all_scores:
            continue
        s = all_scores[model_key]
        label = "qwen2.5:7b (base)" if model_key == "base" else "**fred:latest (ours)** *"
        lines.append(f"| {label} | {s['numerical']['recall']} | {s['temporal']['recall']} "
                     f"| {s['overall']['recall']} | {s['binary']['recall']} |")

    # ── Editing Metrics ──────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 4. Editing Accuracy (Deterministic)")
    lines.append("")
    lines.append("| Metric | qwen2.5:7b (base) | fred:latest (ours) |")
    lines.append("|---|---|---|")

    metrics_labels = [
        ("XML Validity Rate", "editing", "xml_valid_rate"),
        ("Wrong-Span (Delete) Accuracy", "editing", "delete_accuracy"),
        ("Corrected-Span (Mark) Accuracy", "editing", "mark_accuracy"),
        ("Complete Edit Success", "editing", "complete_edit_success"),
        ("Clean Preservation Rate", "clean", "rate"),
    ]

    for label, group, key in metrics_labels:
        base_val = all_scores.get("base", {}).get(group, {}).get(key, "—")
        fred_val = all_scores.get("fred", {}).get(group, {}).get(key, "—")
        lines.append(f"| {label} | {base_val}% | {fred_val}% |")

    # ── Binary Confusion Matrix ──────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 5. Binary Detection Confusion Matrix")
    for model_key in ["base", "fred"]:
        if model_key not in all_scores:
            continue
        s = all_scores[model_key]
        b = s["binary"]
        label = "qwen2.5:7b (base)" if model_key == "base" else "fred:latest (ours)"
        lines.append(f"\n### {label}")
        lines.append("")
        lines.append("| | Predicted Error | Predicted Clean |")
        lines.append("|---|---|---|")
        lines.append(f"| **Actual Error** | TP={b['tp']} | FN={b['fn']} |")
        lines.append(f"| **Actual Clean** | FP={b['fp']} | TN={b['tn']} |")

    # ── Detailed Counts ──────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 6. Detailed Counts")
    lines.append("")
    for model_key in ["base", "fred"]:
        if model_key not in all_scores:
            continue
        s = all_scores[model_key]
        label = "qwen2.5:7b (base)" if model_key == "base" else "fred:latest (ours)"
        lines.append(f"\n### {label}")
        lines.append(f"- Edit attempts (GT=error & model=error): {s['editing']['edit_attempts']}")
        lines.append(f"- XML valid: {s['editing']['xml_valid_count']}/{s['editing']['xml_total_with_tags']}")
        lines.append(f"- Delete correct: {s['editing']['delete_correct']}/{s['editing']['edit_attempts']}")
        lines.append(f"- Mark correct: {s['editing']['mark_correct']}/{s['editing']['edit_attempts']}")
        lines.append(f"- Complete edit success: {s['editing']['ces_count']}/{s['editing']['edit_attempts']}")
        lines.append(f"- Clean preserved: {s['clean']['preserved']}/{s['clean']['total']}")

    # ── Sample Failures ──────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 7. Sample Failures (fred:latest, up to 10)")
    lines.append("")
    if "fred" in all_scores:
        for f in all_scores["fred"]["sample_failures"][:10]:
            lines.append(f"- `{f.get('source_id', '?')}`: "
                         f"GT={f.get('gt_type','?')}, Pred={f.get('pd_type','?')}, "
                         f"Del=[{f.get('gt_delete','?')} vs {f.get('pd_delete','?')}], "
                         f"Mark=[{f.get('gt_mark','?')} vs {f.get('pd_mark','?')}], "
                         f"XML={f.get('xml_valid','?')}, "
                         f"Reason={f.get('reason','')}")
    else:
        lines.append("No fred predictions available.")

    # ── FactScore Placeholder ────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 8. Editing Results — FActScoreLite (Paper Table 5 Format)")
    lines.append("")
    lines.append("> **TODO**: Run LLM-as-judge evaluation using Gemini/DeepSeek/Groq.")
    lines.append("> This section will be appended after running the FactScore evaluation.")
    lines.append("")
    lines.append("| Model | Editor (LLM Judge) | Score |")
    lines.append("|---|---|---|")
    lines.append("| No Edit (baseline) | — | — |")
    lines.append("| qwen2.5:7b (base) | — | — |")
    lines.append("| fred:latest (ours) | — | — |")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("FRED EVALUATION — SCORING")
    print("=" * 72)

    all_scores = {}

    for model_key, pred_path in PRED_FILES.items():
        if not pred_path.exists():
            print(f"\n  [{model_key}] Predictions file not found: {pred_path}")
            print(f"  Run scripts/12_run_evaluation.py first.")
            continue

        print(f"\n  [{model_key}] Loading predictions from {pred_path} ...")
        predictions = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))

        print(f"  [{model_key}] Scoring {len(predictions)} predictions ...")
        scores = score_model(predictions)
        all_scores[model_key] = scores

        print(f"  [{model_key}] Results:")
        print(f"    Binary F1:     {scores['binary']['f1']}")
        print(f"    Numerical F1:  {scores['numerical']['f1']}")
        print(f"    Temporal F1:   {scores['temporal']['f1']}")
        print(f"    Overall F1:    {scores['overall']['f1']}")
        print(f"    CES:           {scores['editing']['complete_edit_success']}%")
        print(f"    Clean Pres.:   {scores['clean']['rate']}%")

    # Save scores
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        # Convert for JSON serialization
        json.dump(all_scores, f, indent=2, default=str)
    print(f"\n  Scores saved to {SCORES_PATH}")

    # Generate report
    report = generate_report(all_scores)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to {REPORT_PATH}")

    print(f"\n{'=' * 72}")
    print("SCORING COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
