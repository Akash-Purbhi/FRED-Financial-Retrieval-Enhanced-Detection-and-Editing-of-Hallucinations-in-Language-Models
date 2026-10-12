#!/usr/bin/env python3
"""
10_build_hybrid_dataset.py
===========================
Build the final hybrid FRED training dataset offline.

Sources:
  - 447 canonical-valid Gemini rows (data/audit/high_confidence_canonical_valid.jsonl)
  - Official FinQA ground truth (data/official_finqa/{train,dev,test}.json)
  - RagBench FinQA natural-language responses (rungalileo/ragbench, finqa)

Produces:
  - data/final/{train,validation,test}.jsonl
  - data/final/{train,validation,test}_chatml.jsonl
  - data/final/rejected.jsonl
  - reports/final_dataset_report.md
"""

import json
import math
import random
import re
import sys
import copy
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

# ── paths ────────────────────────────────────────────────────────────────────
WORKSPACE     = Path(__file__).resolve().parent.parent
GEMINI_VALID  = WORKSPACE / "data" / "audit" / "high_confidence_canonical_valid.jsonl"
OFFICIAL_DIR  = WORKSPACE / "data" / "official_finqa"
FINAL_DIR     = WORKSPACE / "data" / "final"
REPORT_DIR    = WORKSPACE / "reports"
REPORT_OUT    = REPORT_DIR / "final_dataset_report.md"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ── constants ────────────────────────────────────────────────────────────────
TARGET_NUM_CORRUPTED_MIN  = 4000
TARGET_NUM_CORRUPTED_MAX  = 4300
TARGET_TEMPORAL_MIN       = 250
TARGET_TEMPORAL_MAX       = 500
TARGET_CLEAN_MIN          = 1000
TARGET_CLEAN_MAX          = 1250

CORRUPTION_METHODS = [
    "digit_substitution",
    "small_additive",
    "percentage_perturbation",
    "decimal_perturbation",
    "ratio_perturbation",
    "rounded_value",
]

TEMPORAL_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
TEMPORAL_QUARTER_RE = re.compile(
    r"\b(Q[1-4]|first quarter|second quarter|third quarter|fourth quarter"
    r"|1st quarter|2nd quarter|3rd quarter|4th quarter)\b", re.I
)
TEMPORAL_FISCAL_RE = re.compile(
    r"\b(fiscal\s*(?:year\s*)?(?:19[89]\d|20[0-3]\d)|fy\s*(?:19[89]\d|20[0-3]\d))\b", re.I
)
TEMPORAL_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b", re.I
)

# ── helpers ──────────────────────────────────────────────────────────────────

def norm_q(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def norm_number(val: str) -> str:
    v = val.strip().replace(",", "").replace("$", "").rstrip("%").strip()
    try:
        return f"{float(v):g}"
    except ValueError:
        return v.lower().strip()


def format_answer(exe_ans, text_ans: str) -> str:
    """Return the best human-readable answer string."""
    if text_ans and text_ans.lower() not in ("", "nan"):
        return text_ans
    return str(exe_ans)


def find_answer_span(response: str, text_ans: str, exe_ans) -> tuple:
    """Find the canonical answer span in the response.

    Returns (start_idx, end_idx, matched_text) or None.
    Tries progressively looser match strategies.
    """
    if not response:
        return None

    # Strategy 1: exact text_ans match
    if text_ans and text_ans in response:
        idx = response.index(text_ans)
        return (idx, idx + len(text_ans), text_ans)

    # Strategy 2: exe_ans as-is
    exe_str = str(exe_ans)
    if exe_str in response:
        idx = response.index(exe_str)
        return (idx, idx + len(exe_str), exe_str)

    # Strategy 3: formatted variants
    candidates = _generate_answer_variants(text_ans, exe_ans)
    for candidate in candidates:
        if candidate and candidate in response:
            idx = response.index(candidate)
            return (idx, idx + len(candidate), candidate)

    # Strategy 4: regex for the numeric value with optional formatting
    try:
        if isinstance(exe_ans, float):
            # Try various decimal precisions
            for prec in range(0, 6):
                formatted = f"{exe_ans:.{prec}f}"
                if formatted in response:
                    idx = response.index(formatted)
                    return (idx, idx + len(formatted), formatted)
            # Try with comma separators
            if abs(exe_ans) >= 1000:
                formatted = f"{exe_ans:,.0f}"
                if formatted in response:
                    idx = response.index(formatted)
                    return (idx, idx + len(formatted), formatted)
                formatted = f"{exe_ans:,.2f}"
                if formatted in response:
                    idx = response.index(formatted)
                    return (idx, idx + len(formatted), formatted)
    except (ValueError, TypeError):
        pass

    return None


def _generate_answer_variants(text_ans: str, exe_ans) -> list:
    """Generate formatted variants of the answer to search in response."""
    variants = []
    if not text_ans and exe_ans is None:
        return variants

    # From text_ans
    if text_ans:
        # With/without $ sign
        variants.append(f"${text_ans}")
        variants.append(text_ans.replace("$", "").strip())
        # With/without % sign
        if "%" in text_ans:
            variants.append(text_ans.replace("%", "").strip())
        else:
            variants.append(f"{text_ans}%")
        # Comma variants
        variants.append(text_ans.replace(",", ""))

    # From exe_ans
    try:
        val = float(exe_ans)
    except (ValueError, TypeError):
        return [v for v in variants if v]

    # Percentage equivalence: 0.25 <-> 25%
    if abs(val) < 1 and val != 0:
        pct = val * 100
        for prec in range(0, 4):
            s = f"{pct:.{prec}f}%"
            variants.append(s)
            variants.append(s.replace("%", ""))
            variants.append(f"{pct:.{prec}f}")

    # Dollar formatting
    if abs(val) >= 1:
        for prec in [0, 1, 2]:
            variants.append(f"${val:,.{prec}f}")
            variants.append(f"{val:,.{prec}f}")
            variants.append(f"${val:.{prec}f}")
            variants.append(f"{val:.{prec}f}")

    # Integer formatting
    if val == int(val):
        iv = int(val)
        variants.append(str(iv))
        variants.append(f"{iv:,}")
        variants.append(f"${iv:,}")

    return [v for v in variants if v]


# ── corruption functions ─────────────────────────────────────────────────────

def corrupt_digit_substitution(value_str: str, value_float: float) -> str:
    """Change exactly one digit in the numeric string."""
    digits = [(i, c) for i, c in enumerate(value_str) if c.isdigit()]
    if not digits:
        return None
    for _ in range(20):
        idx, orig_digit = random.choice(digits)
        new_digit = random.choice([d for d in "0123456789" if d != orig_digit])
        candidate = value_str[:idx] + new_digit + value_str[idx+1:]
        # Don't create leading zeros for multi-digit numbers
        clean = candidate.lstrip("$").lstrip("-").lstrip()
        if clean and clean[0] == "0" and len(clean) > 1 and clean[1] != ".":
            continue
        if candidate != value_str:
            return candidate
    return None


def corrupt_small_additive(value_str: str, value_float: float) -> str:
    """Add or subtract a small amount."""
    if value_float == 0:
        delta = random.choice([1, -1, 0.1, -0.1, 0.5, -0.5])
    else:
        magnitude = abs(value_float)
        delta_pct = random.uniform(0.01, 0.15)
        delta = magnitude * delta_pct * random.choice([1, -1])
    new_val = value_float + delta
    return _format_like(new_val, value_str)


def corrupt_percentage_perturbation(value_str: str, value_float: float) -> str:
    """Multiply by a factor near 1.0."""
    factor = random.choice([0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2])
    new_val = value_float * factor
    return _format_like(new_val, value_str)


def corrupt_decimal_perturbation(value_str: str, value_float: float) -> str:
    """Shift decimal point or change precision."""
    if value_float == 0:
        return None
    shift = random.choice([10, 0.1])
    new_val = value_float * shift
    return _format_like(new_val, value_str)


def corrupt_ratio_perturbation(value_str: str, value_float: float) -> str:
    """Swap numerator/denominator for ratio-like values or invert."""
    if value_float == 0:
        return None
    if abs(value_float) < 1 and value_float != 0:
        new_val = 1.0 / value_float
    else:
        new_val = value_float + random.choice([-2, -1, 1, 2]) * (abs(value_float) * 0.1)
    return _format_like(new_val, value_str)


def corrupt_rounded_value(value_str: str, value_float: float) -> str:
    """Round to a different precision or nearby round number."""
    if value_float == 0:
        return corrupt_small_additive(value_str, value_float)
    magnitude = abs(value_float)
    if magnitude >= 100:
        rounding = random.choice([10, 50, 100])
        new_val = round(value_float / rounding) * rounding + random.choice([-rounding, rounding])
    elif magnitude >= 1:
        new_val = round(value_float) + random.choice([-1, 1])
    else:
        new_val = round(value_float, random.choice([1, 2])) + random.choice([-0.01, 0.01, -0.1, 0.1])
    return _format_like(new_val, value_str)


def _format_like(new_val: float, original_str: str) -> str:
    """Format new_val to match the style of original_str (currency, %, commas)."""
    has_dollar = "$" in original_str
    has_pct = "%" in original_str
    has_comma = "," in original_str

    # Determine decimal places in original
    clean = original_str.replace("$", "").replace("%", "").replace(",", "").strip()
    if clean.startswith("-"):
        clean = clean[1:]
    if clean.startswith("(") and clean.endswith(")"):
        clean = clean[1:-1]

    if "." in clean:
        dec_places = len(clean.split(".")[-1])
    else:
        dec_places = 0

    # Format
    if dec_places > 0:
        formatted = f"{new_val:.{dec_places}f}"
    else:
        formatted = f"{int(round(new_val))}"

    # Add commas
    if has_comma:
        parts = formatted.split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        int_part = f"{int(int_part):,}"
        if negative:
            int_part = "-" + int_part
        formatted = int_part + ("." + parts[1] if len(parts) > 1 else "")

    # Add prefix/suffix
    prefix = ""
    suffix = ""
    if has_dollar:
        prefix = "$"
    if has_pct:
        suffix = "%"

    # Check if original had negative formatting
    orig_stripped = original_str.strip()
    if orig_stripped.startswith("-") or orig_stripped.startswith("($"):
        if new_val < 0:
            formatted = formatted.lstrip("-")
            prefix = "-" + prefix

    result = prefix + formatted + suffix
    return result if result != original_str else None


CORRUPTION_FN = {
    "digit_substitution": corrupt_digit_substitution,
    "small_additive": corrupt_small_additive,
    "percentage_perturbation": corrupt_percentage_perturbation,
    "decimal_perturbation": corrupt_decimal_perturbation,
    "ratio_perturbation": corrupt_ratio_perturbation,
    "rounded_value": corrupt_rounded_value,
}


def apply_numerical_corruption(answer_span: str, exe_ans) -> tuple:
    """Apply a random corruption method.
    Returns (corrupted_span, method_name) or (None, None)."""
    try:
        clean = answer_span.replace("$", "").replace("%", "").replace(",", "").strip()
        value = float(clean)
    except ValueError:
        return None, None

    methods = list(CORRUPTION_METHODS)
    random.shuffle(methods)

    for method in methods:
        fn = CORRUPTION_FN[method]
        result = fn(answer_span, value)
        if result and result != answer_span:
            return result, method

    return None, None


# ── temporal corruption ──────────────────────────────────────────────────────

def find_temporal_spans(text: str) -> list:
    """Find all temporal spans (years, quarters, etc.) in text."""
    spans = []
    for m in TEMPORAL_YEAR_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(), "year"))
    for m in TEMPORAL_QUARTER_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(), "quarter"))
    for m in TEMPORAL_MONTH_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(), "month"))
    for m in TEMPORAL_FISCAL_RE.finditer(text):
        spans.append((m.start(), m.end(), m.group(), "fiscal"))
    return spans


def find_temporal_in_context(context: str) -> set:
    """Find all unique temporal values in the context."""
    values = set()
    for m in TEMPORAL_YEAR_RE.finditer(context):
        values.add(m.group())
    for m in TEMPORAL_QUARTER_RE.finditer(context):
        values.add(m.group())
    for m in TEMPORAL_MONTH_RE.finditer(context):
        values.add(m.group())
    return values


def apply_temporal_corruption(response: str, context: str) -> tuple:
    """Find a temporal span in the response that can be swapped with another
    plausible temporal value from the context.

    Returns (corrupted_response, correct_span, wrong_span, method) or (None,)*4.
    """
    resp_spans = find_temporal_spans(response)
    if not resp_spans:
        return None, None, None, None

    ctx_values = find_temporal_in_context(context)
    if not ctx_values:
        return None, None, None, None

    # Shuffle spans for variety
    random.shuffle(resp_spans)

    for start, end, correct_span, span_type in resp_spans:
        # Also check that correct_span is verifiable from context
        if correct_span not in context:
            continue

        # Find candidates from context of the same type
        if span_type == "year":
            candidates = [v for v in ctx_values
                          if TEMPORAL_YEAR_RE.fullmatch(v) and v != correct_span]
        elif span_type == "quarter":
            candidates = [v for v in ctx_values
                          if TEMPORAL_QUARTER_RE.fullmatch(v) and v != correct_span]
        elif span_type == "month":
            candidates = [v for v in ctx_values
                          if TEMPORAL_MONTH_RE.fullmatch(v) and v != correct_span]
        else:
            candidates = [v for v in ctx_values if v != correct_span]

        if not candidates:
            # For years, try adjacent years
            if span_type == "year":
                try:
                    yr = int(correct_span)
                    for delta in [-1, 1, -2, 2]:
                        alt = str(yr + delta)
                        if alt in context and alt != correct_span:
                            candidates.append(alt)
                except ValueError:
                    pass

        if not candidates:
            continue

        wrong_span = random.choice(candidates)

        # Apply the swap at this specific location only
        corrupted = response[:start] + wrong_span + response[end:]

        # Verify exactly one change
        if corrupted == response:
            continue

        return corrupted, correct_span, wrong_span, "temporal_context_swap"

    return None, None, None, None


# ── target construction ──────────────────────────────────────────────────────

def build_target_corrupted(corrupted_response: str, wrong_span: str,
                           correct_span: str, error_type: str) -> str:
    """Build the target_output for a corrupted example."""
    tag = error_type  # "numerical" or "temporal"
    xml_block = f"<{tag}><delete>{wrong_span}</delete><mark>{correct_span}</mark></{tag}>"

    # Replace the FIRST occurrence of wrong_span in corrupted_response with the XML block
    idx = corrupted_response.find(wrong_span)
    if idx == -1:
        return None

    tagged_response = (
        corrupted_response[:idx]
        + xml_block
        + corrupted_response[idx + len(wrong_span):]
    )

    return f"Correction: {tagged_response}"


def build_target_clean(clean_response: str) -> str:
    """Build the target_output for a clean example."""
    return f"Correction: {clean_response}"


# ── inline validation helpers ────────────────────────────────────────────────

OUTER_BLOCK_RE = re.compile(r"<(numerical|temporal)>.*?</\1>", re.I | re.DOTALL)

def _build_target_at_position(corrupted_response: str, edit_pos: int,
                               wrong_span: str, correct_span: str,
                               error_type: str) -> str:
    """Build target_output by inserting XML tags at the exact edit position,
    avoiding ambiguity when the wrong_span appears multiple times."""
    tag = error_type
    xml_block = f"<{tag}><delete>{wrong_span}</delete><mark>{correct_span}</mark></{tag}>"

    # Verify wrong_span starts at edit_pos in corrupted_response
    actual = corrupted_response[edit_pos:edit_pos + len(wrong_span)]
    if actual != wrong_span:
        # Fall back to first-occurrence search
        idx = corrupted_response.find(wrong_span)
        if idx == -1:
            return None
        edit_pos = idx

    tagged = (
        corrupted_response[:edit_pos]
        + xml_block
        + corrupted_response[edit_pos + len(wrong_span):]
    )
    return f"Correction: {tagged}"


def _inline_validate(target: str, corrupted_response: str,
                     source_response: str, wrong_span: str,
                     correct_span: str, error_type: str) -> bool:
    """Run the same checks the post-build validator uses, inline.
    Returns True if the example passes all checks."""

    # 1. Extract correction text
    prefix = "Correction: "
    if target.startswith(prefix):
        corr_text = target[len(prefix):]
    else:
        return False

    # 2. Find the outer XML block
    m = OUTER_BLOCK_RE.search(corr_text)
    if not m:
        return False

    # 3. Reconstruction check: replace XML block with correct_span
    #    -> should equal source_response
    reconstructed_correct = corr_text[:m.start()] + correct_span + corr_text[m.end():]
    if reconstructed_correct.strip() != source_response.strip():
        return False

    # 4. Reconstruction check: replace XML block with wrong_span
    #    -> should equal corrupted_response
    reconstructed_wrong = corr_text[:m.start()] + wrong_span + corr_text[m.end():]
    if reconstructed_wrong.strip() != corrupted_response.strip():
        return False

    # 5. Residual wrong span check: after applying correction,
    #    wrong_span should not remain near the edit location
    if wrong_span not in correct_span:
        edit_start = m.start()
        edit_end = edit_start + len(correct_span)
        region = reconstructed_correct[max(0, edit_start - 5):edit_end + 5]
        if wrong_span in region:
            return False

    return True



def load_official_finqa() -> dict:
    """Load official FinQA splits. Returns dict[norm_question] -> (split, item)."""
    index = {}
    ambiguous_qs = set()
    for split in ("train", "dev", "test"):
        path = OFFICIAL_DIR / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            q = norm_q(item["qa"]["question"])
            if q in index:
                ambiguous_qs.add(q)
            else:
                index[q] = (split, item)
    # Remove ambiguous questions
    for q in ambiguous_qs:
        if q in index:
            del index[q]
    print(f"  Official FinQA: {len(index)} unique questions ({len(ambiguous_qs)} ambiguous removed)")
    return index


def load_ragbench() -> dict:
    """Load RagBench responses. Returns dict[ragbench_id] -> record."""
    from datasets import load_dataset
    ds = load_dataset("rungalileo/ragbench", "finqa")
    index = {}
    for split in ds:
        for r in ds[split]:
            index[r["id"]] = {
                "question": r["question"],
                "response": r["response"],
                "documents": r["documents"],
            }
    print(f"  RagBench: {len(index)} records loaded")
    return index


def load_gemini_valid() -> list:
    """Load the 447 canonical-valid Gemini rows."""
    rows = []
    with open(GEMINI_VALID, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"  Gemini valid: {len(rows)} rows")
    return rows


# ── source document formatting ───────────────────────────────────────────────

def format_source_documents(official_item: dict) -> str:
    """Format official FinQA pre_text + table + post_text as source_documents."""
    parts = []
    pre = " ".join(official_item.get("pre_text", []))
    if pre.strip():
        parts.append(pre.strip())

    table = official_item.get("table", [])
    if table:
        table_str = json.dumps(table)
        parts.append(table_str)

    post = " ".join(official_item.get("post_text", []))
    if post.strip():
        parts.append(post.strip())

    return json.dumps(parts)


# ── main build pipeline ──────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("HYBRID DATASET BUILDER")
    print("=" * 72)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n[1/8] Loading data sources ...")
    official_index = load_official_finqa()
    ragbench = load_ragbench()
    gemini_valid = load_gemini_valid()

    # ── Step 2: Index used source IDs ─────────────────────────────────────
    print("\n[2/8] Indexing Gemini seed rows ...")
    gemini_source_ids = set()
    gemini_questions = set()
    for row in gemini_valid:
        sid = row.get("source_id", "")
        gemini_source_ids.add(sid)
        gemini_questions.add(norm_q(row.get("question", "")))
    print(f"  {len(gemini_source_ids)} Gemini source IDs reserved")

    # ── Step 3: Build eligible source pool ────────────────────────────────
    print("\n[3/8] Building eligible source pool ...")
    eligible = []
    skip_reasons = Counter()

    for rb_id, rb_data in ragbench.items():
        if rb_id in gemini_source_ids:
            skip_reasons["already_used_by_gemini"] += 1
            continue

        q_norm = norm_q(rb_data["question"])
        if q_norm not in official_index:
            skip_reasons["no_official_match"] += 1
            continue

        off_split, off_item = official_index[q_norm]
        qa = off_item["qa"]
        exe_ans = qa.get("exe_ans", "")
        text_ans = str(qa.get("answer", ""))
        program = str(qa.get("program", ""))

        # Must have valid program and answer
        if not program or program == "nan":
            skip_reasons["no_program"] += 1
            continue

        # Skip yes/no answers (not suitable for numerical span replacement)
        if str(exe_ans).lower() in ("yes", "no", "true", "false"):
            skip_reasons["yes_no_answer"] += 1
            continue

        response = rb_data["response"]
        if not response or len(response.strip()) < 10:
            skip_reasons["empty_response"] += 1
            continue

        # Build source documents
        source_docs = format_source_documents(off_item)

        eligible.append({
            "source_id": rb_id,
            "official_split": off_split,
            "question": rb_data["question"],
            "source_documents": source_docs,
            "source_response": response,
            "canonical_answer": text_ans,
            "canonical_exe_ans": exe_ans,
            "canonical_program": program,
            "context_text": " ".join(off_item.get("pre_text", []))
                          + " " + json.dumps(off_item.get("table", []))
                          + " " + " ".join(off_item.get("post_text", [])),
        })

    print(f"  Eligible sources: {len(eligible)}")
    print(f"  Skip reasons: {dict(skip_reasons.most_common(10))}")

    # ── Step 4: Generate temporal corruptions ─────────────────────────────
    print("\n[4/8] Generating temporal corruptions ...")
    temporal_examples = []
    temporal_rejected = []
    random.shuffle(eligible)

    for src in eligible:
        if len(temporal_examples) >= TARGET_TEMPORAL_MAX:
            break

        response = src["source_response"]
        context = src["context_text"]

        corrupted, correct_span, wrong_span, method = apply_temporal_corruption(
            response, context
        )
        if corrupted is None:
            continue

        target = build_target_corrupted(corrupted, wrong_span, correct_span, "temporal")
        if target is None:
            temporal_rejected.append({"source_id": src["source_id"], "reason": "target_build_failed"})
            continue

        # Validate: wrong_span in corrupted, correct_span != wrong_span
        if wrong_span not in corrupted:
            temporal_rejected.append({"source_id": src["source_id"], "reason": "wrong_span_not_in_corrupted"})
            continue
        if wrong_span == correct_span:
            temporal_rejected.append({"source_id": src["source_id"], "reason": "spans_identical"})
            continue

        # Inline validation: reconstruction check
        if not _inline_validate(target, corrupted, response, wrong_span, correct_span, "temporal"):
            temporal_rejected.append({"source_id": src["source_id"], "reason": "inline_validation_failed"})
            continue

        temporal_examples.append({
            "source_id": src["source_id"],
            "official_split": src["official_split"],
            "question": src["question"],
            "source_documents": src["source_documents"],
            "source_response": src["source_response"],
            "response": corrupted,
            "target_output": target,
            "error_type": "temporal",
            "canonical_answer": src["canonical_answer"],
            "canonical_program": src["canonical_program"],
            "corruption_method": method,
            "data_origin": "deterministic",
            "is_clean": False,
        })

    print(f"  Temporal corrupted: {len(temporal_examples)}")
    print(f"  Temporal rejected: {len(temporal_rejected)}")
    temporal_ids = set(ex["source_id"] for ex in temporal_examples)

    # ── Step 5: Generate numerical corruptions ────────────────────────────
    print("\n[5/8] Generating numerical corruptions ...")
    numerical_examples = []
    numerical_rejected = []

    # Determine how many numerical we need
    num_target = TARGET_NUM_CORRUPTED_MAX - len(temporal_examples)
    print(f"  Target numerical: ~{num_target}")

    # Re-shuffle and go through eligible sources
    random.shuffle(eligible)
    for src in eligible:
        if len(numerical_examples) >= num_target:
            break

        if src["source_id"] in temporal_ids:
            continue  # Don't reuse temporal sources for numerical

        response = src["source_response"]
        text_ans = src["canonical_answer"]
        exe_ans = src["canonical_exe_ans"]

        # Find the answer span in the response
        span_result = find_answer_span(response, text_ans, exe_ans)
        if span_result is None:
            numerical_rejected.append({
                "source_id": src["source_id"],
                "reason": "answer_not_found_in_response",
            })
            continue

        start_idx, end_idx, matched_span = span_result

        # Apply numerical corruption
        corrupted_span, method = apply_numerical_corruption(matched_span, exe_ans)
        if corrupted_span is None:
            numerical_rejected.append({
                "source_id": src["source_id"],
                "reason": "corruption_failed",
            })
            continue

        # Build corrupted response by replacing at the EXACT location
        corrupted_response = response[:start_idx] + corrupted_span + response[end_idx:]

        # Verify the corrupted span actually differs
        if corrupted_response == response:
            numerical_rejected.append({
                "source_id": src["source_id"],
                "reason": "no_change_after_corruption",
            })
            continue

        # Verify corrupted_span appears in corrupted_response and
        # that we can find it at the same position for targeting
        if corrupted_span not in corrupted_response:
            numerical_rejected.append({
                "source_id": src["source_id"],
                "reason": "corrupted_span_not_in_response",
            })
            continue

        # Build target using the EXACT position to avoid multi-occurrence issues
        target = _build_target_at_position(
            corrupted_response, start_idx, corrupted_span, matched_span, "numerical"
        )
        if target is None:
            numerical_rejected.append({
                "source_id": src["source_id"],
                "reason": "target_build_failed",
            })
            continue

        # Inline validation: reconstruction and residual checks
        if not _inline_validate(target, corrupted_response, response, corrupted_span, matched_span, "numerical"):
            numerical_rejected.append({
                "source_id": src["source_id"],
                "reason": "inline_validation_failed",
            })
            continue

        numerical_examples.append({
            "source_id": src["source_id"],
            "official_split": src["official_split"],
            "question": src["question"],
            "source_documents": src["source_documents"],
            "source_response": src["source_response"],
            "response": corrupted_response,
            "target_output": target,
            "error_type": "numerical",
            "canonical_answer": src["canonical_answer"],
            "canonical_program": src["canonical_program"],
            "corruption_method": method,
            "data_origin": "deterministic",
            "is_clean": False,
        })

    print(f"  Numerical corrupted: {len(numerical_examples)}")
    print(f"  Numerical rejected: {len(numerical_rejected)}")
    numerical_ids = set(ex["source_id"] for ex in numerical_examples)

    # ── Step 6: Prepare Gemini rows with full metadata ────────────────────
    print("\n[6/8] Preparing Gemini rows with full metadata ...")
    gemini_examples = []
    for row in gemini_valid:
        sid = row.get("source_id", "")
        q_norm = norm_q(row.get("question", ""))

        # Get official data
        if q_norm in official_index:
            off_split, off_item = official_index[q_norm]
        else:
            off_split = "train"
            off_item = None

        # Get RagBench data for source_response
        rb = ragbench.get(sid, {})
        source_response = rb.get("response", "")

        # Get source_documents
        if off_item:
            source_docs = format_source_documents(off_item)
            canon_ans = str(off_item["qa"].get("answer", ""))
            canon_prog = str(off_item["qa"].get("program", ""))
        else:
            source_docs = json.dumps(rb.get("documents", []))
            canon_ans = ""
            canon_prog = ""

        # Determine error_type from target_output
        target = row.get("target_output", "")
        if "<temporal>" in target.lower():
            error_type = "temporal"
        else:
            error_type = "numerical"

        gemini_examples.append({
            "source_id": sid,
            "official_split": off_split,
            "question": row.get("question", ""),
            "source_documents": source_docs,
            "source_response": source_response,
            "response": row.get("response", ""),
            "target_output": target,
            "error_type": error_type,
            "canonical_answer": canon_ans,
            "canonical_program": canon_prog,
            "corruption_method": "gemini_llm",
            "data_origin": "gemini",
            "is_clean": False,
        })

    print(f"  Gemini examples prepared: {len(gemini_examples)}")

    # ── Step 7: Generate clean examples ───────────────────────────────────
    print("\n[7/8] Generating clean examples ...")
    used_ids = gemini_source_ids | temporal_ids | numerical_ids
    clean_eligible = [s for s in eligible if s["source_id"] not in used_ids]
    random.shuffle(clean_eligible)

    clean_examples = []
    for src in clean_eligible:
        if len(clean_examples) >= TARGET_CLEAN_MAX:
            break

        response = src["source_response"]
        if not response or len(response.strip()) < 10:
            continue

        target = build_target_clean(response)

        clean_examples.append({
            "source_id": src["source_id"],
            "official_split": src["official_split"],
            "question": src["question"],
            "source_documents": src["source_documents"],
            "source_response": response,
            "response": response,  # clean: response IS the correct response
            "target_output": target,
            "error_type": "none",
            "canonical_answer": src["canonical_answer"],
            "canonical_program": src["canonical_program"],
            "corruption_method": "none",
            "data_origin": "deterministic",
            "is_clean": True,
        })

    print(f"  Clean examples: {len(clean_examples)}")

    # ── Step 8: Split, write, report ──────────────────────────────────────
    print("\n[8/8] Splitting and writing outputs ...")

    # Merge all examples
    all_examples = gemini_examples + numerical_examples + temporal_examples + clean_examples

    # Map official split -> dataset split
    split_map = {"train": "train", "dev": "validation", "test": "test"}
    splits = defaultdict(list)
    for ex in all_examples:
        ds_split = split_map.get(ex["official_split"], "train")
        splits[ds_split].append(ex)

    # Verify no source leakage
    leakage_check = defaultdict(set)
    for split_name, rows in splits.items():
        for r in rows:
            leakage_check[r["source_id"]].add(split_name)
    leaked_sources = {sid: s for sid, s in leakage_check.items() if len(s) > 1}
    if leaked_sources:
        print(f"  WARNING: {len(leaked_sources)} sources appear in multiple splits!")
    else:
        print("  No source leakage detected.")

    # Check for source_id reuse in both corrupted and clean within same partition
    for split_name, rows in splits.items():
        corrupted_ids = set(r["source_id"] for r in rows if not r["is_clean"])
        clean_ids = set(r["source_id"] for r in rows if r["is_clean"])
        overlap = corrupted_ids & clean_ids
        if overlap:
            print(f"  WARNING: {len(overlap)} sources in both corrupted and clean in {split_name}")

    # Write outputs
    all_rejected = []
    for r in numerical_rejected + temporal_rejected:
        all_rejected.append(r)

    for split_name in ["train", "validation", "test"]:
        rows = splits.get(split_name, [])
        random.shuffle(rows)

        # Write raw JSONL
        out_path = FINAL_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {out_path.name}: {len(rows)} rows")

        # Write ChatML
        chatml_path = FINAL_DIR / f"{split_name}_chatml.jsonl"
        with open(chatml_path, "w", encoding="utf-8") as f:
            for r in rows:
                chatml = _to_chatml(r)
                f.write(json.dumps(chatml, ensure_ascii=False) + "\n")
        print(f"  {chatml_path.name}: {len(rows)} rows")

    # Write rejected
    rejected_path = FINAL_DIR / "rejected.jsonl"
    with open(rejected_path, "w", encoding="utf-8") as f:
        for r in all_rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  rejected.jsonl: {len(all_rejected)} rows")

    # ── Generate report ──────────────────────────────────────────────────
    _generate_report(splits, all_examples, gemini_examples, numerical_examples,
                     temporal_examples, clean_examples, all_rejected,
                     leaked_sources, skip_reasons)

    print(f"\n  Report: {REPORT_OUT}")
    print("\n" + "=" * 72)
    print("DATASET BUILD COMPLETE")
    print("=" * 72)


def _to_chatml(record: dict) -> dict:
    """Convert a record to ChatML format."""
    # Build context from source_documents
    docs = record.get("source_documents", "")
    try:
        doc_list = json.loads(docs) if isinstance(docs, str) else docs
    except (json.JSONDecodeError, TypeError):
        doc_list = [docs] if docs else []

    # Format documents
    context_parts = []
    for i, d in enumerate(doc_list[:3]):
        d_str = str(d).strip()
        if len(d_str) > 600:
            d_str = d_str[:600] + "..."
        context_parts.append(d_str)
    context = "\n\n".join(context_parts)

    question = record.get("question", "")
    response = record.get("response", "")
    target = record.get("target_output", "")

    user_content = f"Context: {context}\nQuestion: {question}\nResponse: {response}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ]
    }


def _generate_report(splits, all_examples, gemini_examples, numerical_examples,
                     temporal_examples, clean_examples, rejected,
                     leaked_sources, skip_reasons):
    """Generate the final dataset report."""
    total = len(all_examples)
    total_corrupted = len(gemini_examples) + len(numerical_examples) + len(temporal_examples)
    total_clean = len(clean_examples)

    # Token length analysis (approximate: 1 token ~ 4 chars)
    def approx_tokens(record):
        chatml = _to_chatml(record)
        text = chatml["messages"][0]["content"] + chatml["messages"][1]["content"]
        return len(text) // 4

    token_lengths = [approx_tokens(r) for r in all_examples]
    token_lengths.sort()

    if token_lengths:
        max_tokens = max(token_lengths)
        p50 = token_lengths[len(token_lengths) // 2]
        p90 = token_lengths[int(len(token_lengths) * 0.9)]
        p95 = token_lengths[int(len(token_lengths) * 0.95)]
        p99 = token_lengths[int(len(token_lengths) * 0.99)]
    else:
        max_tokens = p50 = p90 = p95 = p99 = 0

    # Duplicate source_ids
    all_ids = [r["source_id"] for r in all_examples]
    id_counts = Counter(all_ids)
    duplicates = {k: v for k, v in id_counts.items() if v > 1}

    # Corruption method distribution
    method_dist = Counter(r["corruption_method"] for r in all_examples if not r["is_clean"])

    # Split stats
    split_stats = {}
    for split_name in ["train", "validation", "test"]:
        rows = splits.get(split_name, [])
        corrupted = [r for r in rows if not r["is_clean"]]
        clean = [r for r in rows if r["is_clean"]]
        gemini = [r for r in rows if r["data_origin"] == "gemini"]
        deterministic = [r for r in rows if r["data_origin"] == "deterministic"]
        numerical = [r for r in rows if r["error_type"] == "numerical"]
        temporal = [r for r in rows if r["error_type"] == "temporal"]
        split_stats[split_name] = {
            "total": len(rows),
            "corrupted": len(corrupted),
            "clean": len(clean),
            "gemini": len(gemini),
            "deterministic": len(deterministic),
            "numerical": len(numerical),
            "temporal": len(temporal),
        }

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("# Final Hybrid Dataset Report\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

        f.write("## 1. Dataset Summary\n\n")
        f.write("| Metric | Count |\n")
        f.write("|---|---|\n")
        f.write(f"| **Total examples** | **{total}** |\n")
        f.write(f"| Total corrupted | {total_corrupted} |\n")
        f.write(f"| Total clean | {total_clean} |\n")
        f.write(f"| Numerical | {len(numerical_examples) + sum(1 for g in gemini_examples if g['error_type']=='numerical')} |\n")
        f.write(f"| Temporal | {len(temporal_examples) + sum(1 for g in gemini_examples if g['error_type']=='temporal')} |\n")
        f.write(f"| Gemini (LLM-generated) | {len(gemini_examples)} |\n")
        f.write(f"| Deterministic (Python) | {len(numerical_examples) + len(temporal_examples)} |\n")
        f.write(f"| Rejected during generation | {len(rejected)} |\n\n")

        f.write("## 2. Split Distribution\n\n")
        f.write("| Split | Total | Corrupted | Clean | Gemini | Deterministic | Numerical | Temporal |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for split_name in ["train", "validation", "test"]:
            s = split_stats.get(split_name, {})
            f.write(f"| {split_name} | {s.get('total',0)} | {s.get('corrupted',0)} | "
                    f"{s.get('clean',0)} | {s.get('gemini',0)} | {s.get('deterministic',0)} | "
                    f"{s.get('numerical',0)} | {s.get('temporal',0)} |\n")
        f.write("\n")

        f.write("## 3. Corruption Method Distribution\n\n")
        f.write("| Method | Count |\n")
        f.write("|---|---|\n")
        for method, count in method_dist.most_common():
            f.write(f"| {method} | {count} |\n")
        f.write("\n")

        f.write("## 4. Token Length Analysis (approx)\n\n")
        f.write("| Metric | Tokens |\n")
        f.write("|---|---|\n")
        f.write(f"| Maximum | {max_tokens} |\n")
        f.write(f"| P50 (median) | {p50} |\n")
        f.write(f"| P90 | {p90} |\n")
        f.write(f"| P95 | {p95} |\n")
        f.write(f"| P99 | {p99} |\n\n")

        f.write("## 5. Duplicate Source IDs\n\n")
        if duplicates:
            f.write(f"**{len(duplicates)} duplicate source IDs found:**\n\n")
            for sid, count in list(sorted(duplicates.items(), key=lambda x: -x[1]))[:20]:
                f.write(f"- `{sid}`: {count} occurrences\n")
        else:
            f.write("No duplicate source IDs.\n")
        f.write("\n")

        f.write("## 6. Source Leakage Check\n\n")
        if leaked_sources:
            f.write(f"**WARNING: {len(leaked_sources)} sources appear in multiple splits!**\n\n")
            for sid, s in list(leaked_sources.items())[:10]:
                f.write(f"- `{sid}`: {s}\n")
        else:
            f.write("No source leakage detected across splits.\n")
        f.write("\n")

        f.write("## 7. Rejected Examples\n\n")
        reject_reasons = Counter(r.get("reason", "unknown") for r in rejected)
        f.write("| Reason | Count |\n")
        f.write("|---|---|\n")
        for reason, count in reject_reasons.most_common():
            f.write(f"| {reason} | {count} |\n")
        f.write("\n")

        f.write("## 8. Source Selection Skip Reasons\n\n")
        f.write("| Reason | Count |\n")
        f.write("|---|---|\n")
        for reason, count in skip_reasons.most_common():
            f.write(f"| {reason} | {count} |\n")
        f.write("\n")

        f.write("## 9. Output Files\n\n")
        f.write("| File | Rows |\n")
        f.write("|---|---|\n")
        for split_name in ["train", "validation", "test"]:
            s = split_stats.get(split_name, {})
            f.write(f"| `data/final/{split_name}.jsonl` | {s.get('total',0)} |\n")
            f.write(f"| `data/final/{split_name}_chatml.jsonl` | {s.get('total',0)} |\n")
        f.write(f"| `data/final/rejected.jsonl` | {len(rejected)} |\n")


if __name__ == "__main__":
    main()
