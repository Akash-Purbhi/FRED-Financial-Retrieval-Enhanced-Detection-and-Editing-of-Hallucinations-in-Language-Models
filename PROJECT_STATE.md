# FRED — Project State

## Current Objective
Generate remaining synthetic samples to complete 1,500 valid rows on FinQA (939 / 1,500 collected).

---

## Architecture & Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fine-tune model | Qwen3-4B via Unsloth | Efficient 4-bit QLoRA on consumer GPU |
| Datasets | FinQA + TAT-QA (RagBench) | Financial focus; FAVA dropped |
| Baseline eval model | Gemini or Groq | Pending evaluation batch-size decision |
| Error-insertion model | `gemini-3.1-flash-lite` via Google Gemini API | Reliable high-throughput generation with 15 RPM rate limiting |
| Filtering pipeline | Format, non-identical text, consistency checks | FRED paper quality enforcement (`filter_data.py`) |
| Format transformation | ChatML (`chatml_finqa_1500.jsonl`) | Unsloth fine-tuning format (`scripts/04_transform_chatml.py`) |
| API key management | `FRED/.env` + `python-dotenv` | Keeps secrets out of notebooks and git |

---

## Milestones

| # | Milestone | Status | Notebook/File |
|---|---|---|---|
| 1 | Load & inspect FinQA + TAT-QA from RagBench | ✅ Done | `notebooks/01_inspect_datasets.ipynb` |
| 2 | Schema verified + error-insertion notebook created | ✅ Done | `notebooks/02_error_insertion_finqa.ipynb` |
| 3 | `.env` key management wired up | ✅ Done | `.env` + notebook Cell 2 |
| 4 | Run & validate error insertion on FinQA (prototype) | ✅ Done | `notebooks/02_error_insertion_finqa.ipynb` + `groq_utils.py` |
| 5 | Bulk generate 1,500 filtered synthetic samples | 🟡 In Progress (939/1500) | `scripts/02_generate_dataset.py` |
| 6 | Transform dataset into ChatML format | ✅ Done | `scripts/04_transform_chatml.py` |
| 7 | Baseline evaluation | ⬜ Not started | — |
| 8 | Fine-tuning (QLoRA via Unsloth) | ⬜ Not started | — |
| 9 | Post-fine-tune evaluation & comparison | ⬜ Not started | — |

---

## Completed Milestones

- **Datasets loaded** — FinQA and TAT-QA from `rungalileo/ragbench` loaded into Jupyter.
- **Row counts verified:**
  - FinQA : ~16,500 rows
  - TAT-QA: ~33,100 rows
- **Schema inspected** — key fields confirmed: `documents`, `question`, `response`.
- **API key management**: `FRED/.env` created; modules load key via `python-dotenv`. `.env` is in `.gitignore`.
- **Error-insertion prototype complete** (`groq_utils.py`, `notebooks/02_error_insertion_finqa.ipynb`):
  - 6 error types: `temporal`, `numerical`, `entity`, `relation`, `contradictory`, `unverifiable`.
  - Prompt enforces lowercase tag formatting and prohibits `<think>`/CoT blocks.
  - Programmatic safety net strips residual `<think>` blocks.
  - Case-insensitive regex parsing (`re.IGNORECASE`) validates tagged outputs.
  - Architecture refactored into modular `config.py` + `groq_utils.py` with thin notebook wrapper.
- **FRED Data Quality Filter** (`filter_data.py`):
  - Invalid format check, identical delete/mark text check, and content reconstruction consistency check.
- **Dataset Generation (939 / 1,500 samples)**:
  - Generates using `gemini-3.1-flash-lite` with 15 RPM throttling (4.5s delay) and auto-resume.
- **ChatML Transformation Pipeline** (`scripts/04_transform_chatml.py`):
  - Converts synthetic samples into Unsloth ChatML format:
    - User message: `Context`, `Question`, `erroneous_passage` (unwrapped text with hallucinated span).
    - Assistant message: `target_output` (swapped `<delete>` and `<mark>` tags pointing to ground-truth corrections).

---

## Repo Structure

```
FRED/
├── .env                          ← Environment variables (never committed)
├── .gitignore
├── LLM FRED.pdf
├── PROJECT_STATE.md              ← Project tracker
├── config.py                     ← Central configurations
├── groq_utils.py                 ← Prompts and utility helpers
├── filter_data.py                ← FRED 3-stage validation filter
├── notebooks/
│   ├── 01_inspect_datasets.ipynb
│   └── 02_error_insertion_finqa.ipynb
├── scripts/
│   ├── 02_generate_dataset.py    ← Gemini dataset generator with auto-resume
│   ├── 03_generate_dataset_gemini.py
│   └── 04_transform_chatml.py    ← ChatML conversion pipeline
└── synthetic_finqa_1500.jsonl    ← Output dataset (939 rows generated)
```

---

## Active Bugs & Blockers
_None_

---

## Notes
- All notebooks live in `notebooks/` and are numbered sequentially.
- `PROJECT_STATE.md` is updated after every meaningful change.
- Git remote: `https://github.com/Akash-Purbhi/FRED-Financial-Retrieval-Enhanced-Detection-and-Editing-of-Hallucinations-in-Language-Models`
