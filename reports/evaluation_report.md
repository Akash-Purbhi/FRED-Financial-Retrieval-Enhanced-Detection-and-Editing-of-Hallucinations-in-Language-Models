# FRED Evaluation Report

**Generated**: 2026-10-06T00:22:09.652864

**Test set**: 841 records (651 corrupted + 190 clean)

**Models compared**: fred:latest (fine-tuned) vs qwen2.5:7b (base)

---

## 1. Detection Performance (F1-score) — Paper Table 3 Format

| Editor | Num. | Temp. | Overall | Binary |
|---|---|---|---|---|
| GPT-4.1 mini *(paper)* | 23.5 | 24.6 | 46.0 | 77.8 |
| o3 *(paper)* | 52.2 | 83.8 | 71.9 | 90.3 |
| Phi4-mini-8k* *(paper)* | 29.1 | 51.9 | 63.9 | 83.8 |
| QWEN3-4b-36k* *(paper)* | 48.2 | 58.8 | 72.4 | 89.7 |
| Phi4-8k* *(paper)* | 71.4 | 85.7 | 89.9 | 96.7 |
| qwen2.5:7b (base) | 0.0 | 0.0 | 0.0 | 0.0 |
| **fred:latest (ours)** * | 84.5 | 64.7 | 81.6 | 91.2 |

\* fine-tuned model

---

## 2. Detection Performance (Precision) — Paper Table 11 Format

| Editor | Num. | Temp. | Overall | Binary |
|---|---|---|---|---|
| qwen2.5:7b (base) | 0.0 | 0.0 | 0.0 | 0.0 |
| **fred:latest (ours)** * | 79.5 | 57.0 | 76.1 | 85.0 |

---

## 3. Detection Performance (Recall) — Paper Table 12 Format

| Editor | Num. | Temp. | Overall | Binary |
|---|---|---|---|---|
| qwen2.5:7b (base) | 0.0 | 0.0 | 0.0 | 0.0 |
| **fred:latest (ours)** * | 90.1 | 74.7 | 88.0 | 98.3 |

---

## 4. Editing Accuracy (Deterministic)

| Metric | qwen2.5:7b (base) | fred:latest (ours) |
|---|---|---|
| XML Validity Rate | 0.0% | 97.0% |
| Wrong-Span (Delete) Accuracy | 0.0% | 49.7% |
| Corrected-Span (Mark) Accuracy | 0.0% | 48.3% |
| Complete Edit Success | 0.0% | 46.9% |
| Clean Preservation Rate | 100.0% | 40.5% |

---

## 5. Binary Detection Confusion Matrix

### qwen2.5:7b (base)

| | Predicted Error | Predicted Clean |
|---|---|---|
| **Actual Error** | TP=0 | FN=651 |
| **Actual Clean** | FP=0 | TN=190 |

### fred:latest (ours)

| | Predicted Error | Predicted Clean |
|---|---|---|
| **Actual Error** | TP=640 | FN=11 |
| **Actual Clean** | FP=113 | TN=77 |

---

## 6. Detailed Counts


### qwen2.5:7b (base)
- Edit attempts (GT=error & model=error): 0
- XML valid: 0/0
- Delete correct: 0/0
- Mark correct: 0/0
- Complete edit success: 0/0
- Clean preserved: 190/190

### fred:latest (ours)
- Edit attempts (GT=error & model=error): 640
- XML valid: 621/640
- Delete correct: 318/640
- Mark correct: 309/640
- Complete edit success: 300/640
- Clean preserved: 77/190

---

## 7. Sample Failures (fred:latest, up to 10)

- `finqa_6257`: GT=numerical, Pred=numerical, Del=[9 vs 9], Mark=[10 vs 1], XML=True, Reason=
- `finqa_6728`: GT=numerical, Pred=numerical, Del=[25 vs $25], Mark=[45 vs $45], XML=True, Reason=
- `finqa_7229`: GT=numerical, Pred=numerical, Del=[8 vs $253,908], Mark=[7 vs $253,907], XML=True, Reason=
- `finqa_6982`: GT=numerical, Pred=numerical, Del=[14.98% vs None], Mark=[17.32% vs None], XML=False, Reason=
- `finqa_6798`: GT=numerical, Pred=numerical, Del=[12.7% vs 12], Mark=[14.7% vs 14], XML=True, Reason=
- `finqa_6516`: GT=numerical, Pred=numerical, Del=[2 vs 163,2], Mark=[0 vs 163,0], XML=True, Reason=
- `finqa_6604`: GT=numerical, Pred=numerical, Del=[198.0 vs 0.4], Mark=[172.2 vs 1.5], XML=False, Reason=
- `finqa_6633`: GT=numerical, Pred=numerical, Del=[74.1% vs 74], Mark=[44.1% vs 44], XML=True, Reason=
- `finqa_6958`: GT=numerical, Pred=temporal, Del=[3 vs 2013], Mark=[4 vs 2014], XML=True, Reason=
- `finqa_7116`: GT=numerical, Pred=numerical, Del=[2 vs $94], Mark=[3 vs $94], XML=True, Reason=

---

## 8. Editing Results — FActScoreLite (Paper Table 5 Format)

> **LLM Judge**: `gemini-3.1-flash-lite` | **Samples judged**: 1000 | **Generated**: 2026-10-12T13:28:52.766624

### 8.1 Overall Factual Accuracy (Mean Score 1-5)

| Condition | Mean Score | % Correct (≥4) | % Perfect (5) | N |
|---|---|---|---|---|
| No Edit (corrupted baseline) | 3.022 | 39.2% | 23.4% | 500 |
| **FRED Edit (ours)** | **3.77** | **66.6%** | **63.6%** | 500 |
| **Δ Improvement** | **+0.748** | — | — | — |

### 8.2 Factual Accuracy by Error Type (Corrupted Samples Only)

| Error Type | No Edit (Mean) | FRED Edit (Mean) | Δ |
|---|---|---|---|
| Numerical | 2.709 | 3.817 | +1.108 |
| Temporal | 2.917 | 3.688 | +0.771 |

### 8.3 Clean Sample Preservation

| Condition | Mean Score (Clean) | % Perfect (5) | N |
|---|---|---|---|
| No Edit | 4.065 | 73.1% | 108 |
| FRED Edit | 3.657 | 60.2% | 108 |

### 8.4 Score Distribution

| Score | No Edit | FRED Edit |
|---|---|---|
| 1 ⭐ | 133 | 126 |
| 2 ⭐⭐ | 36 | 14 |
| 3 ⭐⭐⭐ | 135 | 27 |
| 4 ⭐⭐⭐⭐ | 79 | 15 |
| 5 ⭐⭐⭐⭐⭐ | 117 | 318 |