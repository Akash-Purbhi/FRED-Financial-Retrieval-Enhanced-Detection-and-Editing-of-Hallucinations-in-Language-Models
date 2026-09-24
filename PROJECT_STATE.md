# FRED — Project State

## Current Objective
Evaluate and optimize inference-time prompting strategies for the fine-tuned FRED model. Considering retraining with 3-4 epochs to strengthen XML tagging patterns.

---

## Architecture & Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fine-tune model | Qwen2.5-7B-Instruct via Unsloth | Efficient 4-bit QLoRA on consumer GPU |
| Quantized model | `Qwen2.5-7B-Instruct.Q4_K_M.gguf` | GGUF format for Ollama local inference |
| Datasets | FinQA + TAT-QA (RagBench) | Financial focus; FAVA dropped |
| Baseline eval model | Gemini or Groq | Pending evaluation batch-size decision |
| Error-insertion model | `gemini-3.1-flash-lite` via Google Gemini API | Reliable high-throughput generation with 15 RPM rate limiting |
| Filtering pipeline | Format, non-identical text, consistency checks | FRED paper quality enforcement (`filter_data.py`) |
| Format transformation | ChatML (`chatml_finqa_1500.jsonl`) | Unsloth fine-tuning format (`scripts/04_transform_chatml.py`) |
| Local inference | Ollama with custom Modelfile | `fred_qwen` model with ChatML template & stop tokens |
| Inference strategy | JSON extraction → XML tagging pipeline | Avoids distribution shift from CoT; `temperature: 0` enforced |
| API key management | `FRED/.env` + `python-dotenv` | Keeps secrets out of notebooks and git |

---

## Milestones

| # | Milestone | Status | Notebook/File |
|---|---|---|---|
| 1 | Load & inspect FinQA + TAT-QA from RagBench | ✅ Done | `notebooks/01_inspect_datasets.ipynb` |
| 2 | Schema verified + error-insertion notebook created | ✅ Done | `notebooks/02_error_insertion_finqa.ipynb` |
| 3 | `.env` key management wired up | ✅ Done | `.env` + notebook Cell 2 |
| 4 | Run & validate error insertion on FinQA (prototype) | ✅ Done | `notebooks/02_error_insertion_finqa.ipynb` + `groq_utils.py` |
| 5 | Bulk generate 1,500 filtered synthetic samples | ✅ Done | `scripts/02_generate_dataset.py` |
| 6 | Transform dataset into ChatML format | ✅ Done | `scripts/04_transform_chatml.py` |
| 7 | Fine-tuning (QLoRA via Unsloth, 2 epochs) | ✅ Done | Unsloth (external — Google Colab) |
| 8 | Ollama deployment with custom Modelfile | ✅ Done | `Modelfile (1)` → `ollama create fred_qwen` |
| 9 | Inference testing — direct prompting | ✅ Done | `fred_router.py` (v1) |
| 10 | Inference testing — CoT prompting | ✅ Done | `fred_router.py` (v2) — fixed $88→$78 but lost XML tags |
| 11 | Inference testing — JSON extraction pipeline | 🟡 In Progress | `fred_router.py` (v3) — structured fallback |
| 12 | Retrain with 3-4 epochs | ⬜ Planned | Unsloth (Google Colab) |
| 13 | Baseline evaluation | ⬜ Not started | — |
| 14 | Post-fine-tune evaluation & comparison | ⬜ Not started | — |

---

## Key Findings (Inference Testing)

| Test | Result | Takeaway |
|---|---|---|
| **Direct few-shot** | Model rushed to output — inconsistent XML tagging | LoRA weights learned tagging but need more epochs to internalize |
| **Chain-of-Thought** | ✅ Correctly identified `$78` from context, but ❌ dropped XML tags entirely | Distribution shift — model trained for direct tagging, not reasoning-then-tagging |
| **JSON extraction pipeline** | 🟡 Testing | Structured extraction avoids distribution shift; Python validates before XML stage |

**Recommendation**: Retrain with **3-4 epochs** (currently 2) to strengthen XML pattern. `temperature: 0` is mandatory at inference.

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
- **Dataset Generation (1,500 samples)** — Complete:
  - Generates using `gemini-3.1-flash-lite` with 15 RPM throttling (4.5s delay) and auto-resume.
- **ChatML Transformation Pipeline** (`scripts/04_transform_chatml.py`):
  - Converts synthetic samples into Unsloth ChatML format:
    - User message: `Context`, `Question`, `erroneous_passage` (unwrapped text with hallucinated span).
    - Assistant message: `target_output` (swapped `<delete>` and `<mark>` tags pointing to ground-truth corrections).
- **Fine-tuning completed** — 2 epochs via Unsloth QLoRA on Google Colab. Exported as GGUF (Q4_K_M).
- **Ollama deployment** — Custom `Modelfile` with ChatML template, `temperature: 0.1`, stop tokens for `<|im_end|>` and `<|endoftext|>`.
- **Inference testing** — 3 prompting strategies tested (direct, CoT, JSON pipeline).

---

## Repo Structure

```
FRED/
├── .env                          ← Environment variables (never committed)
├── .gitignore
├── LLM FRED.pdf
├── Modelfile (1)                 ← Ollama model config (template, params, system prompt)
├── PROJECT_STATE.md              ← Project tracker
├── config.py                     ← Central configurations
├── groq_utils.py                 ← Prompts and utility helpers
├── filter_data.py                ← FRED 3-stage validation filter
├── fred_router.py                ← Inference pipeline (JSON extraction + XML tagging)
├── notebooks/
│   ├── 01_inspect_datasets.ipynb
│   └── 02_error_insertion_finqa.ipynb
├── scripts/
│   ├── 02_generate_dataset.py    ← Gemini dataset generator with auto-resume
│   ├── 03_generate_dataset_gemini.py
│   └── 04_transform_chatml.py    ← ChatML conversion pipeline
├── chatml_finqa_1500.jsonl       ← ChatML training dataset (1,500 rows)
├── synthetic_finqa_1500.jsonl    ← Raw synthetic dataset (1,500 rows)
└── Qwen2.5-7B-Instruct.Q4_K_M.gguf  ← GGUF model (git-ignored, 4.6GB)
```

---

## Active Bugs & Blockers

- **XML tag dropout at inference**: Model sometimes produces corrections without XML wrapping. Root cause: likely underfitting at 2 epochs. Planned fix: retrain at 3-4 epochs.
- **Distribution shift with CoT prompting**: CoT forces reasoning the model wasn't trained for, breaking XML output. Mitigated by JSON extraction pipeline.

---

## Notes
- All notebooks live in `notebooks/` and are numbered sequentially.
- `PROJECT_STATE.md` is updated after every meaningful change.
- GGUF model files are git-ignored (too large for GitHub). Download from HuggingFace or export via Unsloth.
- Git remote: `https://github.com/Akash-Purbhi/FRED-Financial-Retrieval-Enhanced-Detection-and-Editing-of-Hallucinations-in-Language-Models`
