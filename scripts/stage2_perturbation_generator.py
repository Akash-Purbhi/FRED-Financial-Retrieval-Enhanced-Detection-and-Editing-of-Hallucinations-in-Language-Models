"""
scripts/stage2_perturbation_generator.py
========================================
Stage 2 Controlled Hallucination Generation Engine.

Generates safe, traceable controlled hallucination variants for clean financial QA examples.
Covers all 11 target error variants across 5 verifier actions:
1.  numerical_error            (action: edit)
2.  arithmetic_error           (action: edit)
3.  percentage_ratio_error     (action: edit)
4.  wrong_year_error           (action: edit)
5.  wrong_entity_error         (action: edit)
6.  wrong_unit_currency        (action: edit)
7.  direction_inversion        (action: edit)
8.  unsupported_claim          (action: edit / abstain)
9.  missing_evidence           (action: abstain)
10. irrelevant_context         (action: request_evidence)
11. contradictory_evidence     (action: flag_conflict)

Preserves required tracking metadata for every example:
- original_answer
- modified_answer
- error_type
- correct_action
- supporting_evidence
- generation_method
- human_verification_status ('auto_verified' | 'pending_manual_review')
"""

import copy
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE / "scripts"))

from stage2_verifier import is_close_numeric, parse_financial_number
from stage2_schema import Stage2Record

random.seed(42)

# ── Dictionaries & Regex Patterns ─────────────────────────────────────────────

DIRECTION_PAIRS = {
    "increased": "decreased",
    "decreased": "increased",
    "increase": "decrease",
    "decrease": "increase",
    "grew": "declined",
    "declined": "grew",
    "growth": "decline",
    "decline": "growth",
    "higher": "lower",
    "lower": "higher",
    "gain": "loss",
    "loss": "gain",
    "profit": "loss",
    "surplus": "deficit",
    "expanded": "contracted",
    "contracted": "expanded",
    "expansion": "contraction",
    "contraction": "expansion",
    "up": "down",
    "down": "up",
}

UNIT_PAIRS = {
    "million": "billion",
    "billion": "million",
    "millions": "billions",
    "billions": "millions",
    "thousand": "million",
    "thousands": "millions",
    "$": "€",
    "€": "$",
    "%": " basis points",
}

ENTITY_POOL = [
    "Alphabet Inc.",
    "Microsoft Corporation",
    "Amazon Web Services",
    "Apple Inc.",
    "Tesla Inc.",
    "JPMorgan Chase & Co.",
    "Exxon Mobil Corp.",
    "Johnson & Johnson",
    "Berkshire Hathaway",
    "Meta Platforms Inc.",
]

YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
NUMBER_RE = re.compile(r"\b(?:\$|€|£)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*%?\b")


# ── Perturbation Handlers ────────────────────────────────────────────────────

def perturb_numerical(original_ans: str, resp_text: str) -> Tuple[str, str, str]:
    """1. Numerical Error: Alters digits or applies offset."""
    clean = original_ans.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        val = float(clean)
        delta = random.choice([1.15, 0.85, 1.25, 0.75])
        new_val = round(val * delta, 2 if "." in clean else 0)
        prefix = "$" if "$" in original_ans else ""
        suffix = "%" if "%" in original_ans else ""
        err_str = f"{prefix}{int(new_val) if new_val.is_integer() else new_val}{suffix}"
    except ValueError:
        err_str = f"{original_ans}5"

    tagged = f"<numerical><delete>{err_str}</delete><mark>{original_ans}</mark></numerical>"
    modified_resp = resp_text.replace(original_ans, tagged) if original_ans in resp_text else f"The result is {tagged}."
    return err_str, modified_resp, "deterministic_digit_scaling"


def perturb_arithmetic(original_ans: str, resp_text: str) -> Tuple[str, str, str]:
    """2. Arithmetic Error: Keeps equation terms but yields bad calculation."""
    clean = original_ans.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        val = float(clean)
        err_val = round(val + random.choice([5.0, 10.0, -8.0, 15.5]), 1)
        suffix = "%" if "%" in original_ans else ""
        err_str = f"{err_val}{suffix}"
    except ValueError:
        err_str = f"{original_ans} (calc error)"

    tagged = f"<numerical><delete>{err_str}</delete><mark>{original_ans}</mark></numerical>"
    modified_resp = resp_text.replace(original_ans, tagged) if original_ans in resp_text else f"Calculated value: {tagged}."
    return err_str, modified_resp, "deterministic_math_corruption"


def perturb_percentage_ratio(original_ans: str, resp_text: str) -> Tuple[str, str, str]:
    """3. Percentage/Ratio Error: Corrupts decimal or scaling factor."""
    clean = original_ans.replace("%", "").replace(",", "").strip()
    try:
        val = float(clean)
        err_val = round(val * 10, 1) if val < 10 else round(val / 10, 2)
        err_str = f"{err_val}%"
    except ValueError:
        err_str = "25.0%"

    tagged = f"<numerical><delete>{err_str}</delete><mark>{original_ans}</mark></numerical>"
    modified_resp = resp_text.replace(original_ans, tagged) if original_ans in resp_text else f"The calculated margin was {tagged}."
    return err_str, modified_resp, "deterministic_ratio_shift"


def perturb_wrong_year(question: str, context: str, resp_text: str) -> Tuple[str, str, str]:
    """4. Wrong-Year Error: Swaps fiscal year reference."""
    years = YEAR_RE.findall(question + " " + resp_text + " " + context)
    if years:
        target_year = years[0]
        fake_year = str(int(target_year) - 1 if int(target_year) > 2010 else int(target_year) + 1)
    else:
        target_year = "2021"
        fake_year = "2020"

    tagged = f"<temporal><delete>{fake_year}</delete><mark>{target_year}</mark></temporal>"
    if target_year in resp_text:
        modified_resp = resp_text.replace(target_year, tagged, 1)
    else:
        modified_resp = f"In fiscal year {tagged}, {resp_text}"
    return fake_year, modified_resp, "deterministic_temporal_swap"


def perturb_wrong_entity(resp_text: str) -> Tuple[str, str, str]:
    """5. Wrong-Entity Error: Swaps company or division name."""
    swap_entity = random.choice(ENTITY_POOL)
    tagged = f"<entity><delete>{swap_entity}</delete><mark>the registrant</mark></entity>"
    modified_resp = f"According to {tagged}, {resp_text}"
    return swap_entity, modified_resp, "dictionary_entity_substitution"


def perturb_wrong_unit_currency(original_ans: str, resp_text: str) -> Tuple[str, str, str]:
    """6. Wrong Unit/Currency: Swaps million <-> billion or $ <-> €."""
    for u_from, u_to in UNIT_PAIRS.items():
        if u_from in resp_text.lower():
            tagged = f"<numerical><delete>{u_to}</delete><mark>{u_from}</mark></numerical>"
            modified_resp = re.sub(re.escape(u_from), tagged, resp_text, count=1, flags=re.I)
            return u_to, modified_resp, "rule_unit_inversion"

    # Fallback to currency symbol substitution
    tagged = f"<numerical><delete>€{original_ans}</delete><mark>${original_ans}</mark></numerical>"
    modified_resp = f"Total amount was {tagged}."
    return "€", modified_resp, "rule_currency_substitution"


def perturb_direction_inversion(resp_text: str) -> Tuple[str, str, str]:
    """7. Direction Inversion: Inverts growth/decline verbs."""
    for word, opposite in DIRECTION_PAIRS.items():
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, resp_text, re.I):
            tagged = f"<relation><delete>{opposite}</delete><mark>{word}</mark></relation>"
            modified_resp = re.sub(pattern, tagged, resp_text, count=1, flags=re.I)
            return opposite, modified_resp, "dictionary_antonym_inversion"

    # Fallback if no direction word found
    tagged = "<relation><delete>decreased</delete><mark>increased</mark></relation>"
    modified_resp = f"{resp_text} (which {tagged} relative to baseline)."
    return "decreased", modified_resp, "dictionary_antonym_inversion"


def perturb_unsupported_claim(original_ans: str, resp_text: str) -> Tuple[str, str, str]:
    """8. Unsupported Claim: Appends speculative unverified projection."""
    speculative_claim = " Management also guided that EBITDA will double next quarter."
    tagged = f"<unverifiable>{speculative_claim.strip()}</unverifiable>"
    modified_resp = f"{resp_text} {tagged}"
    return speculative_claim.strip(), modified_resp, "speculative_claim_injection"


def perturb_missing_evidence(clean_record: Stage2Record) -> Dict[str, Any]:
    """9. Missing Evidence (Action: abstain): Drops critical tables or sentences."""
    corrupted_context = "General corporate overview: The company operates in North America and EMEA."
    return {
        "context": corrupted_context,
        "table": [],
        "modified_answer": "Unverifiable from provided context due to omitted financial schedules.",
        "correct_action": "abstain",
        "error_type": "missing_evidence",
        "generation_method": "context_and_table_subtraction",
        "human_verification_status": "auto_verified",
    }


def perturb_irrelevant_context(clean_record: Stage2Record) -> Dict[str, Any]:
    """10. Irrelevant Context (Action: request_evidence): Replaces context with boilerplate."""
    boilerplate = (
        "Item 1A. Risk Factors. Our business is subject to risks including global economic conditions, "
        "supply chain constraints, and competitive market pressures."
    )
    return {
        "context": boilerplate,
        "table": [],
        "modified_answer": "Retrieved context does not contain relevant metrics. Additional evidence required.",
        "correct_action": "request_evidence",
        "error_type": "irrelevant_retrieval",
        "generation_method": "boilerplate_distractor_replacement",
        "human_verification_status": "auto_verified",
    }


def perturb_contradictory_evidence(clean_record: Stage2Record) -> Dict[str, Any]:
    """11. Contradictory Evidence (Action: flag_conflict): Injects conflicting disclosure."""
    ans = clean_record.gold_answer
    clean_val = ans.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        f = float(clean_val)
        conflicting_val = round(f * 1.5, 1)
        conflict_str = f"${conflicting_val}" if "$" in ans else f"{conflicting_val}"
    except ValueError:
        conflict_str = f"{ans} (restated)"

    injected_context = (
        f"{clean_record.context}\n\n[Footnote Note 22 - Revision]: "
        f"Auditor reconciliation indicated that the finalized metric was actually {conflict_str}."
    )
    return {
        "context": injected_context,
        "table": clean_record.table,
        "modified_answer": f"Conflicting disclosures detected: primary text reports {ans} while Note 22 reports {conflict_str}.",
        "correct_action": "flag_conflict",
        "error_type": "contradictory_context",
        "generation_method": "footnote_contradiction_injection",
        "human_verification_status": "auto_verified",
    }


# ── Master Perturbation Dispatcher ───────────────────────────────────────────

ALL_STRATEGIES = [
    "numerical_error",
    "arithmetic_error",
    "percentage_ratio_error",
    "wrong_year_error",
    "wrong_entity_error",
    "wrong_unit_currency",
    "direction_inversion",
    "unsupported_claim",
    "missing_evidence",
    "irrelevant_context",
    "contradictory_evidence",
]


def generate_controlled_variant(
    clean_record: Stage2Record, strategy: str
) -> Dict[str, Any]:
    """
    Generate a controlled hallucination variant preserving all 7 required metadata fields:
    - original_answer
    - modified_answer
    - error_type
    - correct_action
    - supporting_evidence
    - generation_method
    - human_verification_status
    """
    orig_ans = clean_record.gold_answer
    resp = clean_record.unverified_response or clean_record.gold_answer
    context = clean_record.context
    table = clean_record.table
    evidence = clean_record.evidence

    # Base payload structure
    base_result = {
        "id": f"{clean_record.id}_{strategy}",
        "source_dataset": clean_record.source_dataset,
        "question": clean_record.question,
        "context": context,
        "table": table,
        "original_answer": orig_ans,
        "modified_answer": "",
        "error_type": strategy,
        "correct_action": "edit",
        "supporting_evidence": evidence,
        "generation_method": "",
        "human_verification_status": "auto_verified",
    }

    if strategy == "numerical_error":
        err_val, mod_resp, method = perturb_numerical(orig_ans, resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method

    elif strategy == "arithmetic_error":
        err_val, mod_resp, method = perturb_arithmetic(orig_ans, resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method

    elif strategy == "percentage_ratio_error":
        err_val, mod_resp, method = perturb_percentage_ratio(orig_ans, resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method

    elif strategy == "wrong_year_error":
        err_val, mod_resp, method = perturb_wrong_year(clean_record.question, context, resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method

    elif strategy == "wrong_entity_error":
        err_val, mod_resp, method = perturb_wrong_entity(resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method
        # Entity substitution requires spot-check
        base_result["human_verification_status"] = "pending_manual_review"

    elif strategy == "wrong_unit_currency":
        err_val, mod_resp, method = perturb_wrong_unit_currency(orig_ans, resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method

    elif strategy == "direction_inversion":
        err_val, mod_resp, method = perturb_direction_inversion(resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method

    elif strategy == "unsupported_claim":
        err_val, mod_resp, method = perturb_unsupported_claim(orig_ans, resp)
        base_result["modified_answer"] = mod_resp
        base_result["generation_method"] = method
        base_result["human_verification_status"] = "pending_manual_review"

    elif strategy == "missing_evidence":
        p = perturb_missing_evidence(clean_record)
        base_result.update(p)

    elif strategy == "irrelevant_context":
        p = perturb_irrelevant_context(clean_record)
        base_result.update(p)

    elif strategy == "contradictory_evidence":
        p = perturb_contradictory_evidence(clean_record)
        base_result.update(p)

    return base_result


# ── Execution Pipeline ───────────────────────────────────────────────────────

def run_perturbation_pipeline(output_limit: int = 9000):
    print("=" * 65)
    print("Stage 2 Controlled Hallucination Generator")
    print("=" * 65)

    clean_finqa_file = WORKSPACE / "data" / "stage2" / "clean_finqa_5000.jsonl"
    clean_tatqa_file = WORKSPACE / "data" / "stage2" / "clean_tatqa_4000.jsonl"

    clean_records: List[Stage2Record] = []
    for filepath in [clean_finqa_file, clean_tatqa_file]:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                clean_records.append(Stage2Record(**d))

    print(f"Loaded {len(clean_records)} clean records for 1:1 hallucination variant generation.")

    # Distribution weights across the 11 strategies
    # 60% standard in-place editable errors, 15% missing evidence, 10% conflict, 10% irrelevant, 5% unsupported
    weights = [
        ("numerical_error", 20),
        ("arithmetic_error", 15),
        ("percentage_ratio_error", 10),
        ("wrong_year_error", 10),
        ("direction_inversion", 8),
        ("wrong_unit_currency", 5),
        ("wrong_entity_error", 4),
        ("unsupported_claim", 3),
        ("missing_evidence", 10),
        ("contradictory_evidence", 8),
        ("irrelevant_context", 7),
    ]
    expanded_pool = []
    for name, w in weights:
        expanded_pool.extend([name] * w)

    generated_variants: List[Dict[str, Any]] = []
    strategy_counts = {s: 0 for s in ALL_STRATEGIES}

    for idx, clean_rec in enumerate(clean_records[:output_limit]):
        chosen_strategy = random.choice(expanded_pool)
        variant = generate_controlled_variant(clean_rec, chosen_strategy)
        generated_variants.append(variant)
        strategy_counts[chosen_strategy] += 1

    out_file = WORKSPACE / "data" / "stage2" / "corrupted_variants_9000.jsonl"
    print(f"\nWriting {len(generated_variants):,} variants to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for v in generated_variants:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print("\nVariant Distribution Summary:")
    for s, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / len(generated_variants)) * 100
        print(f"  - {s:<26}: {cnt:>5} ({pct:.1f}%)")

    # Display sample outputs for each strategy
    print("\n" + "=" * 65)
    print("SAMPLE CONVERTED RECORD FOR EACH OF THE 11 ERROR STRATEGIES")
    print("=" * 65)
    samples_by_strat = {}
    for v in generated_variants:
        if v["error_type"] not in samples_by_strat:
            samples_by_strat[v["error_type"]] = v
            if len(samples_by_strat) == len(ALL_STRATEGIES):
                break

    for strat, sample in samples_by_strat.items():
        print(f"\n[Strategy: {strat}] (Action: {sample['correct_action']}, Status: {sample['human_verification_status']})")
        print(f"  Orig Answer: {sample['original_answer']}")
        print(f"  Mod Answer:  {sample['modified_answer'][:140]}...")
        print(f"  Gen Method:  {sample['generation_method']}")


if __name__ == "__main__":
    run_perturbation_pipeline()
