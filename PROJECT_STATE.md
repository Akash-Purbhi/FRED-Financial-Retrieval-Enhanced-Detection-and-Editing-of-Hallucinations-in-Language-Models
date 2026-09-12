# FRED — Project State

## Current Objective
Run `notebooks/02_error_insertion_finqa.ipynb` and validate that all 5 rows produce correctly tagged corrupted responses.

---

## Architecture & Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fine-tune model | Qwen3-4B via Unsloth | Efficient 4-bit QLoRA on consumer GPU |
| Datasets | FinQA + TAT-QA (RagBench) | Financial focus; FAVA dropped |
| Baseline eval model | Gemini or Groq | Pending evaluation batch-size decision |
| Error-insertion model | `llama-3.1-8b-instant` via Groq API | `gemma2-9b-it` decommissioned by Groq; replaced with Llama 3.1 8B |
| API key management | `FRED/.env` + `python-dotenv` | Keeps secrets out of notebooks and git |

---

## Milestones

| # | Milestone | Status | Notebook/File |
|---|---|---|---|
| 1 | Load & inspect FinQA + TAT-QA from RagBench | ✅ Done | `notebooks/01_inspect_datasets.ipynb` |
| 2 | Schema verified + error-insertion notebook created | ✅ Done | `notebooks/02_error_insertion_finqa.ipynb` |
| 3 | `.env` key management wired up | ✅ Done | `.env` + notebook Cell 2 |
| 4 | Run & validate error insertion on FinQA (first 5 rows) | 🟡 In Progress | `notebooks/02_error_insertion_finqa.ipynb` |
| 5 | Scale error insertion to full FinQA + TAT-QA | ⬜ Not started | — |
| 6 | Baseline evaluation | ⬜ Not started | — |
| 7 | Data preprocessing / prompt formatting | ⬜ Not started | — |
| 8 | Fine-tuning (QLoRA) | ⬜ Not started | — |
| 9 | Post-fine-tune evaluation & comparison | ⬜ Not started | — |

---

## Completed Milestones

- **Datasets loaded** — FinQA and TAT-QA from `rungalileo/ragbench` loaded into Jupyter.
- **Row counts verified:**
  - FinQA : ~16,500 rows
  - TAT-QA: ~33,100 rows
- **Schema inspected** — key fields confirmed: `documents`, `question`, `response`.
- **Error-insertion notebook created** (`02_error_insertion_finqa.ipynb`):
  - Model: `gemma2-9b-it` via Groq
  - 6 error types: Temporal, Numerical, Entity, Relation, Contradictory, Unverifiable
  - Tag format enforced in prompt; regex sanity-check cell validates all outputs
- **API key management**: `FRED/.env` created; notebook loads key via `python-dotenv`.  
  `.env` is in `.gitignore` — key is never committed to GitHub.

---

## Repo Structure

```
FRED/
├── .env                          ← Paste GROQ_API_KEY here (never committed)
├── .gitignore
├── LLM FRED.pdf
├── PROJECT_STATE.md              ← This file
└── notebooks/
    ├── 01_inspect_datasets.ipynb
    └── 02_error_insertion_finqa.ipynb
```

---

## Active Bugs & Blockers
_None_

---

## Notes
- All notebooks live in `notebooks/` and are numbered sequentially.
- `PROJECT_STATE.md` is updated after every meaningful change.
- Git remote: `https://github.com/Akash-Purbhi/FRED-Financial-Retrieval-Enhanced-Detection-and-Editing-of-Hallucinations-in-Language-Models`
