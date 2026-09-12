# FRED — Project State

## Objective
Fine-tune **Qwen3-4B** (via Unsloth) for financial reasoning and document understanding using RagBench data, then evaluate against a Gemini/Groq baseline.

---

## Architecture & Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fine-tune model | Qwen3-4B via Unsloth | Efficient 4-bit QLoRA on consumer GPU |
| Datasets | FinQA + TAT-QA (RagBench) | Financial focus; FAVA dropped |
| Baseline eval model | Gemini or Groq | Pending evaluation batch-size decision |

---

## Milestones

| # | Milestone | Status | Notebook/Script |
|---|---|---|---|
| 1 | Load & inspect FinQA + TAT-QA from RagBench | 🟡 In Progress | `notebooks/01_inspect_datasets.ipynb` |
| 2 | Baseline evaluation | ⬜ Not started | — |
| 3 | Data preprocessing / prompt formatting | ⬜ Not started | — |
| 4 | Fine-tuning (QLoRA) | ⬜ Not started | — |
| 5 | Post-fine-tune evaluation & comparison | ⬜ Not started | — |

---

## Active Bugs & Blockers
_None_

---

## Notes
- All notebooks live in `notebooks/` and are numbered sequentially.
- Scripts (if any) live in `scripts/`.
