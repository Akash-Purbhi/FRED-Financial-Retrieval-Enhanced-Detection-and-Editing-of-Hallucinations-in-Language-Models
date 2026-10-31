# FRED Stage 2 Dataset Generation Report

Generated on: 2026-10-31 23:48:30

## 1. Summary of Generated Splits

| Split | File | Record Count | Clean : Corrupted Ratio |
|---|---|:---:|:---:|
| **Train (Unified JSONL)** | `data/stage2/train.jsonl` | **17,000** | 50% Clean : 50% Corrupted |
| **Train (ChatML Format)** | `data/stage2/train_chatml.jsonl` | **17,000** | Formatted for Qwen 3B / 7B |
| **Validation (JSONL)** | `data/stage2/validation.jsonl` | **1,000** | 500 Clean : 500 Corrupted |
| **Test (Clean Benchmark)** | `data/stage2/test_clean.jsonl` | **1,200** | 100% Untouched Official Ground Truth |
| **Test (Corrupted Suite)** | `data/stage2/test_corrupted.jsonl` | **1,200** | Controlled Hallucination Benchmark |

## 2. Action Distribution (Training Set)

| Action | Count | Percentage |
|---|:---:|:---:|
| `preserve` | 8,500 | 50.0% |
| `edit` | 5,024 | 29.6% |
| `abstain` | 1,738 | 10.2% |
| `flag_conflict` | 904 | 5.3% |
| `request_evidence` | 834 | 4.9% |

## 3. Error Taxonomy Distribution (Training Set)

| Error Type | Count | Percentage |
|---|:---:|:---:|
| `none` | 8,500 | 50.0% |
| `numerical_substitution` | 2,888 | 17.0% |
| `missing_evidence` | 1,738 | 10.2% |
| `temporal_swap` | 1,264 | 7.4% |
| `contradictory_context` | 904 | 5.3% |
| `relation_inversion` | 872 | 5.1% |
| `irrelevant_retrieval` | 834 | 4.9% |
