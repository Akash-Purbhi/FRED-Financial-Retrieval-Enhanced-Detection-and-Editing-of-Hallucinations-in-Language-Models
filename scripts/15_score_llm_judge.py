"""
FRED Evaluation — Step 4: Score LLM Judge Results
===================================================
Aggregates the per-sample LLM judge scores from judge_results.jsonl,
computes mean factual accuracy for each condition (No-Edit vs FRED-Edit),
and updates the evaluation report with Paper Table 5 format results.

Output:
  - data/eval/judge_scores.json       (aggregate scores)
  - reports/evaluation_report.md      (updated with Table 5 results)
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Config ───────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
EVAL_DIR       = ROOT_DIR / "data" / "eval"
JUDGE_PATH     = EVAL_DIR / "judge_results.jsonl"
SCORES_PATH    = EVAL_DIR / "judge_scores.json"
REPORT_PATH    = ROOT_DIR / "reports" / "evaluation_report.md"


# ── Load Results ─────────────────────────────────────────────────────────────

def load_judge_results():
    """Load all judge results."""
    results = []
    with open(JUDGE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj.get("score") is not None:  # skip failures
                    results.append(obj)
    return results


# ── Compute Scores ───────────────────────────────────────────────────────────

def compute_scores(results):
    """Compute aggregate scores grouped by condition and error type."""

    # Group by condition
    by_condition = defaultdict(list)
    # Group by condition + error_type
    by_condition_type = defaultdict(lambda: defaultdict(list))
    # Group by condition + clean/corrupted
    by_condition_status = defaultdict(lambda: {"corrupted": [], "clean": []})

    for r in results:
        cond = r["condition"]
        score = r["score"]
        is_clean = r.get("is_clean", False)
        error_type = r.get("error_type", "none")

        by_condition[cond].append(score)
        by_condition_type[cond][error_type].append(score)

        status_key = "clean" if is_clean else "corrupted"
        by_condition_status[cond][status_key].append(score)

    def stats(scores):
        if not scores:
            return {"mean": 0.0, "count": 0, "dist": {}}
        mean = sum(scores) / len(scores)
        dist = {}
        for s in range(1, 6):
            dist[str(s)] = scores.count(s)
        return {
            "mean": round(mean, 3),
            "count": len(scores),
            "dist": dist,
            "pct_4_or_5": round(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
            "pct_5": round(sum(1 for s in scores if s == 5) / len(scores) * 100, 1),
        }

    output = {
        "generated": datetime.now().isoformat(),
        "total_judgments": len(results),
        "conditions": {},
    }

    for cond in ["no_edit", "fred_edit"]:
        scores = by_condition.get(cond, [])
        cond_data = {
            "overall": stats(scores),
            "by_error_type": {},
            "by_status": {},
        }

        # Per error type
        for etype in ["numerical", "temporal", "none"]:
            type_scores = by_condition_type.get(cond, {}).get(etype, [])
            if type_scores:
                cond_data["by_error_type"][etype] = stats(type_scores)

        # Corrupted vs clean
        for status in ["corrupted", "clean"]:
            status_scores = by_condition_status.get(cond, {}).get(status, [])
            if status_scores:
                cond_data["by_status"][status] = stats(status_scores)

        output["conditions"][cond] = cond_data

    # Compute improvement delta
    no_edit_mean = output["conditions"].get("no_edit", {}).get("overall", {}).get("mean", 0)
    fred_mean = output["conditions"].get("fred_edit", {}).get("overall", {}).get("mean", 0)
    output["delta"] = {
        "mean_improvement": round(fred_mean - no_edit_mean, 3),
        "pct_improvement": round((fred_mean - no_edit_mean) / no_edit_mean * 100, 1) if no_edit_mean > 0 else 0,
    }

    return output


# ── Report Update ────────────────────────────────────────────────────────────

def generate_table5_section(scores):
    """Generate the Table 5 markdown section."""
    lines = []
    lines.append("## 8. Editing Results — FActScoreLite (Paper Table 5 Format)")
    lines.append("")
    lines.append(f"> **LLM Judge**: `{scores.get('judge_model', 'gemini-3.1-flash-lite')}` | "
                 f"**Samples judged**: {scores['total_judgments']} | "
                 f"**Generated**: {scores['generated']}")
    lines.append("")

    no_edit = scores["conditions"].get("no_edit", {}).get("overall", {})
    fred = scores["conditions"].get("fred_edit", {}).get("overall", {})
    delta = scores.get("delta", {})

    # Main comparison table
    lines.append("### 8.1 Overall Factual Accuracy (Mean Score 1-5)")
    lines.append("")
    lines.append("| Condition | Mean Score | % Correct (≥4) | % Perfect (5) | N |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| No Edit (corrupted baseline) | {no_edit.get('mean', '—')} | "
                 f"{no_edit.get('pct_4_or_5', '—')}% | "
                 f"{no_edit.get('pct_5', '—')}% | {no_edit.get('count', '—')} |")
    lines.append(f"| **FRED Edit (ours)** | **{fred.get('mean', '—')}** | "
                 f"**{fred.get('pct_4_or_5', '—')}%** | "
                 f"**{fred.get('pct_5', '—')}%** | {fred.get('count', '—')} |")
    lines.append(f"| **Δ Improvement** | **+{delta.get('mean_improvement', '—')}** | "
                 f"— | — | — |")

    # Per error type breakdown
    lines.append("")
    lines.append("### 8.2 Factual Accuracy by Error Type (Corrupted Samples Only)")
    lines.append("")
    lines.append("| Error Type | No Edit (Mean) | FRED Edit (Mean) | Δ |")
    lines.append("|---|---|---|---|")

    for etype in ["numerical", "temporal"]:
        ne_type = scores["conditions"].get("no_edit", {}).get("by_error_type", {}).get(etype, {})
        fr_type = scores["conditions"].get("fred_edit", {}).get("by_error_type", {}).get(etype, {})
        ne_mean = ne_type.get("mean", "—")
        fr_mean = fr_type.get("mean", "—")
        if isinstance(ne_mean, (int, float)) and isinstance(fr_mean, (int, float)):
            delta_val = f"+{round(fr_mean - ne_mean, 3)}"
        else:
            delta_val = "—"
        lines.append(f"| {etype.capitalize()} | {ne_mean} | {fr_mean} | {delta_val} |")

    # Clean samples performance
    lines.append("")
    lines.append("### 8.3 Clean Sample Preservation")
    lines.append("")
    ne_clean = scores["conditions"].get("no_edit", {}).get("by_status", {}).get("clean", {})
    fr_clean = scores["conditions"].get("fred_edit", {}).get("by_status", {}).get("clean", {})

    lines.append("| Condition | Mean Score (Clean) | % Perfect (5) | N |")
    lines.append("|---|---|---|---|")
    lines.append(f"| No Edit | {ne_clean.get('mean', '—')} | "
                 f"{ne_clean.get('pct_5', '—')}% | {ne_clean.get('count', '—')} |")
    lines.append(f"| FRED Edit | {fr_clean.get('mean', '—')} | "
                 f"{fr_clean.get('pct_5', '—')}% | {fr_clean.get('count', '—')} |")

    # Score distribution
    lines.append("")
    lines.append("### 8.4 Score Distribution")
    lines.append("")
    lines.append("| Score | No Edit | FRED Edit |")
    lines.append("|---|---|---|")
    for s in range(1, 6):
        ne_count = no_edit.get("dist", {}).get(str(s), 0)
        fr_count = fred.get("dist", {}).get(str(s), 0)
        stars = "⭐" * s
        lines.append(f"| {s} {stars} | {ne_count} | {fr_count} |")

    return "\n".join(lines)


def update_report(table5_section):
    """Replace the placeholder Table 5 section in evaluation_report.md."""
    if not REPORT_PATH.exists():
        print(f"  WARNING: Report not found at {REPORT_PATH}. Skipping update.")
        return

    content = REPORT_PATH.read_text(encoding="utf-8")

    # Find and replace the TODO section
    # Pattern: from "## 8. Editing Results" to end of file (or next ## section)
    pattern = r'## 8\. Editing Results.*'
    if re.search(pattern, content, re.DOTALL):
        updated = re.sub(pattern, table5_section, content, flags=re.DOTALL)
        REPORT_PATH.write_text(updated, encoding="utf-8")
        print(f"  Report updated: {REPORT_PATH}")
    else:
        # Append if section not found
        with open(REPORT_PATH, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n" + table5_section + "\n")
        print(f"  Report appended: {REPORT_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("FRED EVALUATION — SCORE LLM JUDGE RESULTS")
    print("=" * 72)

    if not JUDGE_PATH.exists():
        print(f"\n  ERROR: {JUDGE_PATH} not found.")
        print(f"  Run scripts/14_llm_judge.py first.")
        sys.exit(1)

    # Load results
    results = load_judge_results()
    print(f"\n  Loaded {len(results)} valid judgments")

    # Count conditions
    n_no_edit = sum(1 for r in results if r["condition"] == "no_edit")
    n_fred = sum(1 for r in results if r["condition"] == "fred_edit")
    n_failed = sum(1 for r in results if r.get("score") is None)
    print(f"    No-Edit judgments: {n_no_edit}")
    print(f"    FRED-Edit judgments: {n_fred}")
    if n_failed:
        print(f"    Failed (excluded): {n_failed}")

    # Compute scores
    scores = compute_scores(results)
    scores["judge_model"] = "gemini-3.1-flash-lite"

    # Print summary
    print(f"\n{'-' * 40}")
    print("  RESULTS SUMMARY")
    print(f"{'-' * 40}")

    no_edit = scores["conditions"].get("no_edit", {}).get("overall", {})
    fred = scores["conditions"].get("fred_edit", {}).get("overall", {})
    delta = scores.get("delta", {})

    print(f"\n  No-Edit baseline:")
    print(f"    Mean factual accuracy: {no_edit.get('mean', '—')} / 5.0")
    print(f"    % Correct (≥4):       {no_edit.get('pct_4_or_5', '—')}%")
    print(f"    % Perfect (5):        {no_edit.get('pct_5', '—')}%")

    print(f"\n  FRED-Edit (ours):")
    print(f"    Mean factual accuracy: {fred.get('mean', '—')} / 5.0")
    print(f"    % Correct (≥4):       {fred.get('pct_4_or_5', '—')}%")
    print(f"    % Perfect (5):        {fred.get('pct_5', '—')}%")

    print(f"\n  Δ Improvement:          +{delta.get('mean_improvement', '—')}")
    print(f"  % Relative improvement: +{delta.get('pct_improvement', '—')}%")

    # Per error type
    print(f"\n  By Error Type (corrupted only):")
    for etype in ["numerical", "temporal"]:
        ne = scores["conditions"].get("no_edit", {}).get("by_error_type", {}).get(etype, {})
        fr = scores["conditions"].get("fred_edit", {}).get("by_error_type", {}).get(etype, {})
        ne_m = ne.get("mean", "—")
        fr_m = fr.get("mean", "—")
        print(f"    {etype:10s}: No-Edit={ne_m}, FRED={fr_m}")

    # Save scores
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"\n  Scores saved: {SCORES_PATH}")

    # Generate Table 5 section and update report
    table5 = generate_table5_section(scores)
    update_report(table5)

    print(f"\n{'=' * 72}")
    print("SCORING COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
