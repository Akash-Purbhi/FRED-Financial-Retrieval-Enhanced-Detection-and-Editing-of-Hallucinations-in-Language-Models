"""
filter_data.py
Data filtering pipeline for synthetic hallucination insertion
based on the FRED paper methodology.
"""

import re
from typing import Tuple, Optional, Dict, Any

VALID_TAGS = {
    "temporal",
    "numerical",
    "entity",
    "relation",
    "contradictory",
    "unverifiable",
}

TAGS_PATTERN_STR = "|".join(VALID_TAGS)


def validate_format(tagged_response: str) -> Tuple[bool, str, Optional[Dict[str, str]]]:
    """
    Verify that tagged_response contains:
      - Exactly one valid outer error tag pair (<type>...</type>)
      - Inside that tag, exactly one <delete>...</delete> and one <mark>...</mark>
      - No stray or malformed tags anywhere in the response.
    """
    outer_tag_matches = list(
        re.finditer(
            rf"<({TAGS_PATTERN_STR})>(.*?)</\1>",
            tagged_response,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    if len(outer_tag_matches) == 0:
        return False, "Invalid format: No valid outer error tag pair found.", None

    if len(outer_tag_matches) > 1:
        return (
            False,
            f"Invalid format: Expected exactly 1 error tag pair, found {len(outer_tag_matches)}.",
            None,
        )

    match = outer_tag_matches[0]
    tag_name = match.group(1).lower()
    inner_content = match.group(2)

    before_tag = tagged_response[: match.start()]
    after_tag = tagged_response[match.end() :]
    if re.search(rf"</?({TAGS_PATTERN_STR})>", before_tag + after_tag, re.IGNORECASE):
        return False, "Invalid format: Extra or unmatched outer error tags found.", None

    delete_matches = list(
        re.finditer(r"<delete>(.*?)</delete>", inner_content, flags=re.IGNORECASE | re.DOTALL)
    )
    mark_matches = list(
        re.finditer(r"<mark>(.*?)</mark>", inner_content, flags=re.IGNORECASE | re.DOTALL)
    )

    if len(delete_matches) != 1:
        return (
            False,
            f"Invalid format: Expected exactly 1 <delete>...</delete> block, found {len(delete_matches)}.",
            None,
        )

    if len(mark_matches) != 1:
        return (
            False,
            f"Invalid format: Expected exactly 1 <mark>...</mark> block, found {len(mark_matches)}.",
            None,
        )

    all_delete_tags = re.findall(r"</?delete>", inner_content, flags=re.IGNORECASE)
    all_mark_tags = re.findall(r"</?mark>", inner_content, flags=re.IGNORECASE)
    if len(all_delete_tags) != 2:
        return False, "Invalid format: Malformed or unclosed <delete> tag.", None
    if len(all_mark_tags) != 2:
        return False, "Invalid format: Malformed or unclosed <mark> tag.", None

    parsed_info = {
        "tag_name": tag_name,
        "delete_text": delete_matches[0].group(1),
        "mark_text": mark_matches[0].group(1),
    }

    return True, "", parsed_info


def validate_non_identical_text(delete_text: str, mark_text: str) -> Tuple[bool, str]:
    """Check if <delete> and <mark> contents are identical."""
    if delete_text.strip() == mark_text.strip():
        return (
            False,
            f"Identical text: <delete> and <mark> strings are identical ('{delete_text.strip()}').",
        )
    return True, ""


def reconstruct_original_response(tagged_response: str) -> str:
    """Reconstruct original text by stripping mark and unwrapping delete tags."""
    text = re.sub(r"<mark>[\s\S]*?</mark>", "", tagged_response, flags=re.IGNORECASE)
    text = re.sub(r"</?delete>", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"</?(?:{TAGS_PATTERN_STR})>", "", text, flags=re.IGNORECASE)
    return text


def validate_content_consistency(original_response: str, tagged_response: str) -> Tuple[bool, str]:
    """Compare reconstructed response against original response."""
    reconstructed = reconstruct_original_response(tagged_response)
    if reconstructed != original_response:
        return (
            False,
            f"Inconsistent content: Reconstructed text does not match original response.\n"
            f"Expected: '{original_response}'\n"
            f"Got:      '{reconstructed}'",
        )
    return True, ""


def filter_generated_data(original_response: str, tagged_response: str) -> Tuple[bool, str]:
    """
    Validate synthetic hallucination data against FRED quality filters.

    Returns:
      (is_valid, reason)
    """
    format_valid, format_reason, parsed_info = validate_format(tagged_response)
    if not format_valid:
        return False, format_reason

    non_identical, identical_reason = validate_non_identical_text(
        parsed_info["delete_text"], parsed_info["mark_text"]
    )
    if not non_identical:
        return False, identical_reason

    consistent, consistency_reason = validate_content_consistency(
        original_response, tagged_response
    )
    if not consistent:
        return False, consistency_reason

    return True, "Passed all checks"


def run_tests():
    test_cases = [
        {
            "name": "1. Valid Example",
            "original": "Apple reported a net income of $57.4 billion in fiscal year 2020.",
            "tagged": "Apple reported a net income of <numerical><delete>$57.4 billion</delete><mark>$45.2 billion</mark></numerical> in fiscal year 2020.",
            "expect_valid": True,
        },
        {
            "name": "2. Broken Tag Example (missing </delete>)",
            "original": "The operating margin expanded by 150 basis points.",
            "tagged": "The operating margin expanded by <numerical><delete>150 basis points<mark>200 basis points</mark></numerical>.",
            "expect_valid": False,
        },
        {
            "name": "3. Identical Text Example (delete == mark)",
            "original": "Microsoft acquired GitHub in 2018.",
            "tagged": "Microsoft acquired GitHub in <temporal><delete>2018</delete><mark> 2018 </mark></temporal>.",
            "expect_valid": False,
        },
        {
            "name": "4. Inconsistent Content Example (unrelated text modified)",
            "original": "Tesla delivered 499,550 vehicles in 2020, slightly missing its guidance.",
            "tagged": "Tesla delivered <numerical><delete>499,550</delete><mark>500,000</mark></numerical> cars in 2020, beating its guidance.",
            "expect_valid": False,
        },
    ]

    print("\nFRED Synthetic Data Filtering Test Suite\n")

    all_passed = True
    for case in test_cases:
        is_valid, reason = filter_generated_data(case["original"], case["tagged"])
        status = "PASSED" if is_valid == case["expect_valid"] else "FAILED"
        if is_valid != case["expect_valid"]:
            all_passed = False

        print(f"Test Case: {case['name']}")
        print(f"  Valid  : {is_valid} (Expected: {case['expect_valid']}) -> [{status}]")
        print(f"  Reason : {reason}\n")

    summary = "ALL TESTS PASSED [OK]" if all_passed else "SOME TESTS FAILED [FAIL]"
    print(f"Summary: {summary}\n")
    return all_passed


if __name__ == "__main__":
    run_tests()
