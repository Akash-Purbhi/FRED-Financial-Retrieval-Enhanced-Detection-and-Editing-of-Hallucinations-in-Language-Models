# FRED Stage 2: Training Dataset and Evaluation

This document explains what FRED Stage 2 is trained to do, what data is fed to
the model, and how the trained models are evaluated.

## What Stage 2 Learns

Stage 2 performs **evidence-aware financial verification**, rather than only
binary hallucination detection.

For each example, the model receives:

1. Financial context from FinQA or TAT-QA
2. A structured financial table
3. A financial question
4. A candidate answer to verify

It must select one operational action:

| Action | Meaning |
|---|---|
| `preserve` | The answer is correct and should remain unchanged |
| `edit` | The answer contains a correctable factual or numerical error |
| `abstain` | Required evidence is missing |
| `flag_conflict` | Retrieved sources disagree |
| `request_evidence` | Retrieved context is irrelevant |

## 1. Source Data and Schema

The clean examples come from:

- **FinQA**: financial question answering over tables and text
- **TAT-QA**: financial question answering over tables and narrative disclosures

A clean record contains the question, context, table, verified answer, evidence,
reasoning, and the expected action. For example:

```json
{
  "id": "finqa_train_00001",
  "source_dataset": "finqa",
  "question": "What was the profit margin?",
  "context": "The company reported net income of $100 million...",
  "table": [
    ["Metric", "Value"],
    ["Net income", "$100M"],
    ["Revenue", "$500M"]
  ],
  "gold_answer": "20%",
  "evidence": [
    {
      "source_type": "table",
      "location": "[1, 1]",
      "text": "$100M"
    },
    {
      "source_type": "table",
      "location": "[2, 1]",
      "text": "$500M"
    }
  ],
  "reasoning": "100 / 500 × 100 = 20%",
  "operation": "divide(100, 500)",
  "answer_type": "percentage",
  "error_type": "none",
  "action": "preserve"
}
```

The Stage 2 schema captures:

- Record ID
- Source dataset
- Question
- Context
- 2D financial table
- Verified answer
- Grounding evidence
- Reasoning
- Mathematical operation or extraction operation
- Answer type
- Error type
- Correct action

## 2. Controlled Corruption

The pipeline creates traceable hallucination variants from clean financial
records. The original evidence is retained, so the correct answer and required
action are known exactly.

| Error type | Example | Correct action |
|---|---|---|
| Numerical substitution | `$500M` changed to `$450M` | `edit` |
| Temporal swap | FY2023 changed to FY2022 | `edit` |
| Arithmetic error | `20%` changed to `25%` | `edit` |
| Relation inversion | “increased” changed to “decreased” | `edit` |
| Missing evidence | Relevant table removed | `abstain` |
| Contradictory context | A second source reports a different value | `flag_conflict` |
| Irrelevant retrieval | Financial table replaced with unrelated risk text | `request_evidence` |

Each corrupted record tracks:

- Original answer
- Modified answer
- Error type
- Correct action
- Supporting evidence
- Generation method
- Verification status

This makes the labels deterministic and auditable rather than subjective.

## 3. Dataset Composition

### Training set

The training set contains **17,000 records**:

- 8,500 clean records
- 8,500 corrupted records
- Exactly 50% clean and 50% corrupted

Training action distribution:

| Action | Records |
|---|---:|
| `preserve` | 8,500 |
| `edit` | 5,024 |
| `abstain` | 1,738 |
| `flag_conflict` | 904 |
| `request_evidence` | 834 |

The balanced clean/corrupted design addresses the Stage 1 over-editing problem.

### Validation set

The validation set contains **1,000 records**:

- 500 clean
- 500 corrupted

It is used during training to monitor validation loss and generalization. It
does not update the model weights.

### Test sets

Two separate test benchmarks are evaluated after training:

- **1,200 clean records** from untouched official test data
- **1,200 corrupted records** from a controlled hallucination benchmark

## 4. ChatML Training Format

The JSONL records are converted into ChatML messages. A simplified training
example looks like this:

```text
System:
You are FRED Stage 2, an evidence-aware financial hallucination verifier.
Determine whether to preserve, edit, abstain, flag_conflict, or request_evidence.

User:
Context: The company reported net income of $100 million...

Table:
Metric | Value
Net income | $100M
Revenue | $500M

Question:
What was the profit margin?

Response to Verify:
The profit margin was 25%.

Assistant:
{
  "action": "edit",
  "error_type": "arithmetic_miscalculation",
  "operation": "divide(100, 500)",
  "reasoning": "The correct margin is 20%, not 25%.",
  "gold_answer": "20%",
  "tagged_xml": "The profit margin was <numerical><delete>25%</delete><mark>20%</mark></numerical>."
}
```

The model therefore learns to inspect tables, use evidence, verify
calculations, preserve correct answers, abstain when evidence is insufficient,
identify conflicts, and produce structured corrections.

## 5. Fine-Tuning Configuration

For Qwen2.5-3B:

| Setting | Value |
|---|---|
| Base model | Qwen2.5-3B-Instruct |
| Fine-tuning method | LoRA supervised fine-tuning |
| Maximum sequence length | 2,048 tokens |
| Epochs | 2 |
| Learning rate | `2e-4` |
| Per-device batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| LoRA rank | 16 |
| LoRA dropout | 0 |
| Random seed | 42 |

Qwen3-4B uses the same Stage 2 dataset and an almost identical configuration.
The model is shown an input and the correct structured answer, then learns to
reproduce the appropriate action and correction format.

## How Stage 2 Was Tested

### 1. Validation during training

Training uses:

- `train_chatml.jsonl` for learning
- `validation_chatml.jsonl` for validation monitoring

The model must learn to distinguish:

```text
Correct answer        -> preserve
Wrong answer           -> edit
Missing evidence      -> abstain
Conflicting evidence  -> flag_conflict
Irrelevant evidence   -> request_evidence
```

### 2. Clean preservation test

The model is evaluated on 1,200 untouched clean examples.

Example:

```text
Evidence:
Revenue was $500 million.

Candidate answer:
Revenue was $500 million.
```

Expected output:

```json
{
  "action": "preserve"
}
```

Changing a correct answer is a false correction.

```text
CPR = correctly preserved clean answers / total clean answers
```

Reported results:

- Qwen2.5-3B: **99.7% CPR**
- Qwen3-4B: **99.9% CPR**

### 3. Corrupted benchmark test

The model is evaluated separately on 1,200 corrupted examples.

Example:

```text
Evidence:
Revenue was $500 million.

Candidate answer:
Revenue was $450 million.
```

Expected correction:

```xml
Revenue was <numerical><delete>$450 million</delete><mark>$500 million</mark></numerical>.
```

The evaluator checks whether the model:

1. Detects the problem
2. Selects the correct action
3. Identifies the error type
4. Deletes the incorrect span
5. Marks the correct span
6. Produces valid XML
7. Preserves the rest of the answer

### 4. Selective action accuracy

Stage 1 primarily made a binary decision: error or no error. Stage 2 must
distinguish five actions.

For each test record:

```text
Expected action == model action
```

Example:

| Expected action | Model action | Result |
|---|---|---|
| `preserve` | `preserve` | Correct |
| `edit` | `edit` | Correct |
| `abstain` | `edit` | Incorrect |
| `flag_conflict` | `preserve` | Incorrect |
| `request_evidence` | `request_evidence` | Correct |

Qwen2.5-3B achieved **66.5% selective action accuracy**. This is not directly
comparable to Stage 1's 91.2 Binary F1 because Stage 2 is a five-action
classification problem rather than a binary classification problem.

### 5. XML validity

For an edit, the output must contain the required structure:

```xml
<numerical>
  <delete>wrong value</delete>
  <mark>correct value</mark>
</numerical>
```

The evaluator checks whether the tags are structurally valid. Qwen2.5-3B
achieved **58.0% XML validity**.

A structurally valid XML response can still be wrong if it uses incorrect
spans or an incorrect replacement.

### 6. Deleted-span accuracy

This checks whether the model selected the exact incorrect text.

For the candidate answer:

```text
Revenue was $450 million.
```

The expected deleted span is:

```text
$450 million
```

Deleting only `$450`, or deleting the entire sentence, fails strict span
matching. Qwen2.5-3B achieved **57.4% deleted-span accuracy**.

### 7. Marked-span accuracy

This checks whether the model supplied the exact verified replacement.

Expected replacement:

```text
$500 million
```

Depending on strict matching, `$500M`, `$500`, or `$550 million` does not count
as the exact marked span. Qwen2.5-3B achieved **39.8% marked-span accuracy**.

### 8. Strict Complete Edit Success (CES)

CES is the strictest correction metric. The model must:

- Detect the error
- Select `edit`
- Use valid XML
- Delete the exact incorrect span
- Mark the exact correct span
- Preserve the rest of the answer

Successful edit:

```xml
Revenue was <numerical><delete>$450 million</delete><mark>$500 million</mark></numerical>.
```

An output can detect the error but fail CES:

```xml
Revenue was <numerical><delete>$450</delete><mark>$500M</mark></numerical>.
```

Qwen2.5-3B achieved **30.0% strict CES**.

## Reported Results

| Metric | Qwen2.5-3B | Qwen3-4B |
|---|---:|---:|
| Clean preservation rate | **99.7%** | **99.9%** |
| False correction rate | **0.2%** | **0.0%** |
| Selective action accuracy | **66.5%** | Pending |
| XML validity | **58.0%** | Pending |
| Deleted-span accuracy | **57.4%** | Pending |
| Marked-span accuracy | **39.8%** | Pending |
| Strict CES | **30.0%** | Pending |

Additional Qwen2.5-3B results:

- Edit attempts: **814 / 1,200 corrupted examples**
- Clean abstentions: **1**
- Clean test size: **1,200**
- Corrupted test size: **1,200**

