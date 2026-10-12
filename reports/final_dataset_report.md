# Final Hybrid Dataset Report

**Generated**: 2026-10-01T22:34:39.486553

## 1. Dataset Summary

| Metric | Count |
|---|---|
| **Total examples** | **5997** |
| Total corrupted | 4747 |
| Total clean | 1250 |
| Numerical | 4244 |
| Temporal | 503 |
| Gemini (LLM-generated) | 447 |
| Deterministic (Python) | 4300 |
| Rejected during generation | 1444 |

## 2. Split Distribution

| Split | Total | Corrupted | Clean | Gemini | Deterministic | Numerical | Temporal |
|---|---|---|---|---|---|---|---|
| train | 4532 | 3611 | 921 | 447 | 4085 | 3253 | 358 |
| validation | 624 | 485 | 139 | 0 | 624 | 427 | 58 |
| test | 841 | 651 | 190 | 0 | 841 | 564 | 87 |

## 3. Corruption Method Distribution

| Method | Count |
|---|---|
| digit_substitution | 812 |
| rounded_value | 686 |
| decimal_perturbation | 665 |
| small_additive | 562 |
| ratio_perturbation | 559 |
| percentage_perturbation | 516 |
| temporal_context_swap | 500 |
| gemini_llm | 447 |

## 4. Token Length Analysis (approx)

| Metric | Tokens |
|---|---|
| Maximum | 1526 |
| P50 (median) | 655 |
| P90 | 811 |
| P95 | 863 |
| P99 | 971 |

## 5. Duplicate Source IDs

No duplicate source IDs.

## 6. Source Leakage Check

No source leakage detected across splits.

## 7. Rejected Examples

| Reason | Count |
|---|---|
| answer_not_found_in_response | 984 |
| inline_validation_failed | 457 |
| corruption_failed | 3 |

## 8. Source Selection Skip Reasons

| Reason | Count |
|---|---|
| already_used_by_gemini | 447 |
| no_official_match | 305 |
| yes_no_answer | 151 |

## 9. Output Files

| File | Rows |
|---|---|
| `data/final/train.jsonl` | 4532 |
| `data/final/train_chatml.jsonl` | 4532 |
| `data/final/validation.jsonl` | 624 |
| `data/final/validation_chatml.jsonl` | 624 |
| `data/final/test.jsonl` | 841 |
| `data/final/test_chatml.jsonl` | 841 |
| `data/final/rejected.jsonl` | 1444 |


---

## 10. Post-Build Validation Results

**Validation run**: 2026-10-01T22:34:53.247381

### 10.1 Per-Split Validation

| Split | Total | Valid | Invalid | Corrupted | Clean | Gemini | Det. | Numerical | Temporal |
|---|---|---|---|---|---|---|---|---|---|
| train | 4532 | 4532 | 0 | 3611 | 921 | 447 | 4085 | 3253 | 358 |
| validation | 624 | 624 | 0 | 485 | 139 | 0 | 624 | 427 | 58 |
| test | 841 | 841 | 0 | 651 | 190 | 0 | 841 | 564 | 87 |

### 10.2 Failure Reasons

No failures detected.

### 10.3 Source Leakage

No source leakage.

### 10.4 Duplicate Source IDs

No duplicates.

### 10.5 Sample Failures (up to 20)

### 10.6 Overall Status: **PASS**

- Total valid: 5997
- Total invalid: 0
- Source leakage: 0
- Duplicate IDs: 0
