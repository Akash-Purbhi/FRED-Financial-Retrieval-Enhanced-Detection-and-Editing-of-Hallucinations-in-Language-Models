"""
scripts/stage2_verifier.py
===========================
Stage 2 Deterministic Financial Verification Engine.

Performs deterministic mathematical verification for financial calculations:
- Profit, operating, and gross margins
- Period-over-period percentage changes and growth rates
- Financial multiples and ratios (e.g., P/E, debt-to-equity)
- Unit conversions (millions <-> billions, percentage points <-> basis points)
- Relative and absolute numerical tolerance checks
"""

import math
import re
from typing import Optional, Tuple, Dict, Any


def parse_financial_number(val: Any) -> Optional[float]:
    """Extract a clean float from a formatted financial string ($1,234.5M, 15.2%, (45.0))."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace(",", "")
    # Check for accounting negative format: (123.45) -> -123.45
    negative_match = re.match(r"^\((.+)\)$", s)
    if negative_match:
        s = f"-{negative_match.group(1).strip()}"

    # Remove currency symbols and percentage signs
    s = s.replace("$", "").replace("€", "").replace("£", "").replace("%", "").strip()

    # Scale multiplier keywords
    multiplier = 1.0
    if re.search(r"\b(?:billion|bn|b)\b", s, re.I):
        multiplier = 1e9
        s = re.sub(r"\b(?:billion|bn|b)\b", "", s, flags=re.I).strip()
    elif re.search(r"\b(?:million|mn|m)\b", s, re.I):
        multiplier = 1e6
        s = re.sub(r"\b(?:million|mn|m)\b", "", s, flags=re.I).strip()
    elif re.search(r"\b(?:thousand|k)\b", s, re.I):
        multiplier = 1e3
        s = re.sub(r"\b(?:thousand|k)\b", "", s, flags=re.I).strip()

    try:
        return float(s) * multiplier
    except ValueError:
        return None


def is_close_numeric(
    val_a: float, val_b: float, rel_tol: float = 0.005, abs_tol: float = 0.1
) -> bool:
    """
    Check if two numbers are equal within acceptable financial rounding tolerance.
    Allows up to 0.5% relative error or 0.1 absolute error (accounting for 10-K rounding).
    """
    if math.isnan(val_a) or math.isnan(val_b):
        return False
    return math.isclose(val_a, val_b, rel_tol=rel_tol, abs_tol=abs_tol)


def verify_margin(
    income: float, revenue: float, claimed_margin_pct: float, rel_tol: float = 0.005
) -> Dict[str, Any]:
    """Verify profit / operating / gross margin percentage: (income / revenue) * 100."""
    if revenue == 0:
        return {"valid": False, "reason": "Division by zero"}
    expected = (income / revenue) * 100.0
    passed = is_close_numeric(expected, claimed_margin_pct, rel_tol=rel_tol, abs_tol=0.1)
    return {
        "valid": passed,
        "expected": round(expected, 2),
        "claimed": claimed_margin_pct,
        "delta": round(abs(expected - claimed_margin_pct), 4),
        "formula": f"({income} / {revenue}) * 100",
    }


def verify_growth_rate(
    prior: float, current: float, claimed_growth_pct: float, rel_tol: float = 0.005
) -> Dict[str, Any]:
    """Verify percentage change: ((current - prior) / abs(prior)) * 100."""
    if prior == 0:
        return {"valid": False, "reason": "Prior period is zero"}
    expected = ((current - prior) / abs(prior)) * 100.0
    passed = is_close_numeric(expected, claimed_growth_pct, rel_tol=rel_tol, abs_tol=0.1)
    return {
        "valid": passed,
        "expected": round(expected, 2),
        "claimed": claimed_growth_pct,
        "delta": round(abs(expected - claimed_growth_pct), 4),
        "formula": f"(({current} - {prior}) / |{prior}|) * 100",
    }


def verify_ratio(
    numerator: float, denominator: float, claimed_ratio: float, rel_tol: float = 0.005
) -> Dict[str, Any]:
    """Verify multiple or financial ratio: numerator / denominator."""
    if denominator == 0:
        return {"valid": False, "reason": "Denominator is zero"}
    expected = numerator / denominator
    passed = is_close_numeric(expected, claimed_ratio, rel_tol=rel_tol, abs_tol=0.01)
    return {
        "valid": passed,
        "expected": round(expected, 4),
        "claimed": claimed_ratio,
        "delta": round(abs(expected - claimed_ratio), 4),
        "formula": f"{numerator} / {denominator}",
    }


def verify_unit_conversion(
    val: float, from_unit: str, to_unit: str
) -> Optional[float]:
    """Convert between financial reporting units (thousands, millions, billions, basis points)."""
    unit_multipliers = {
        "unit": 1.0,
        "thousand": 1e3,
        "k": 1e3,
        "million": 1e6,
        "m": 1e6,
        "billion": 1e9,
        "b": 1e9,
        "percent": 1.0,
        "bps": 0.01,
        "basis_points": 0.01,
    }
    u_from = from_unit.lower().strip()
    u_to = to_unit.lower().strip()

    if u_from in unit_multipliers and u_to in unit_multipliers:
        base_val = val * unit_multipliers[u_from]
        return base_val / unit_multipliers[u_to]
    return None


if __name__ == "__main__":
    # Self-test routine
    print("Running stage2_verifier self-test...")
    m_test = verify_margin(2707, 40782, 6.638)
    assert m_test["valid"], f"Margin check failed: {m_test}"

    g_test = verify_growth_rate(100, 115, 15.0)
    assert g_test["valid"], f"Growth rate check failed: {g_test}"

    r_test = verify_ratio(500, 250, 2.0)
    assert r_test["valid"], f"Ratio check failed: {r_test}"

    conv = verify_unit_conversion(1.5, "percent", "bps")
    assert conv == 150.0, f"Unit conversion failed: {conv}"

    print("All Stage 2 deterministic financial verification tests passed successfully!")
