# FRED — Project State

## Current Objective
Run and validate the synthetic error-insertion notebook on FinQA (first 5 rows) using Groq `gemma2-9b-it`.

---

## Architecture & Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fine-tune model | Qwen3-4B via Unsloth | Efficient 4-bit QLoRA on consumer GPU |
| Datasets | FinQA + TAT-QA (RagBench) | Financial focus; FAVA dropped |
| Baseline eval model | Gemini or Groq | Pending evaluation batch-size decision |

---

## Milestones

| # | Milestone | Status | Notebook/Script |
|---|---|---|---|
| 1 | Load & inspect FinQA + TAT-QA from RagBench | ✅ Done | `notebooks/01_inspect_datasets.ipynb` |
| 2 | Schema verified + error-insertion notebook created | ✅ Done | `notebooks/02_error_insertion_finqa.ipynb` |
| 3 | Run & validate error insertion on FinQA (first 5 rows) | 🟡 In Progress | `notebooks/02_error_insertion_finqa.ipynb` |
| 4 | Scale error insertion to full FinQA + TAT-QA | ⬜ Not started | — |
| 5 | Baseline evaluation | ⬜ Not started | — |
| 6 | Data preprocessing / prompt formatting | ⬜ Not started | — |
| 7 | Fine-tuning (QLoRA) | ⬜ Not started | — |
| 8 | Post-fine-tune evaluation & comparison | ⬜ Not started | — |

---

## Completed Milestones

- **Datasets successfully loaded into Jupyter.**
- **Row counts verified:**
  - FinQA : ~16,500 rows
  - TAT-QA: ~33,100 rows
- **Schema inspected** — key fields: `documents`, `question`, `response`.
- **Error-insertion notebook created** (`02_error_insertion_finqa.ipynb`)
  - Model: `gemma2-9b-it` via Groq
  - 6 error types: Temporal, Numerical, Entity, Relation, Contradictory, Unverifiable
  - Tag format enforced via prompt; regex sanity check cell included.

---

## Active Bugs & Blockers
_None_

---

## Notes
- All notebooks live in `notebooks/` and are numbered sequentially.
- Scripts (if any) live in `scripts/`.
