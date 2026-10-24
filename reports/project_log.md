# Project Log

## Completed Work
1. **Repository Audit**: Performed non-destructive audits of all generator scripts, grounding metadata, token lengths, and XML syntax.
2. **Dataset Diagnosis**: Evaluated the 5,000-row synthetic CoT FinQA dataset (`synthetic_finqa_cot_5000.jsonl`).
3. **Strict Validation Pipeline**: Engineered a highly precise validation and fuzzy-matching pipeline to discard poorly generated reasoning, malformed XML, and heavily modified facts.
4. **Dataset Scrubbing**: Generated `synthetic_finqa_cot_high_confidence_valid.jsonl`, isolating 1,155 strictly grounded, structurally sound, and fully verifiable records from the initial 5,000 generated samples.
5. **Final Hybrid Dataset Build**: Built `scripts/10_build_hybrid_dataset.py` to deterministically create 5,997 structured Train/Val/Test rows, combining 447 fully canonical-aligned LLM seed records, deterministic Python numerical and temporal corruptions, and 1,250 clean uncorrupted records.
6. **Final Dataset Validation**: Built `scripts/11_validate_hybrid_dataset.py` applying 13 strict checks (source leakage, XML validity, span reconstruction) confirming a 100% pass rate.
7. **Training Notebook Preparation**: Refactored `notebooks/03_kaggle_unsloth_finetune.ipynb` to load the 5,997-row ChatML dataset, configure a Qwen2.5-7B-Instruct 4-bit QLoRA run with response-only loss masking, enforce strict XML formats (without free-form CoT), run smoke tests, and explicitly handle Hugging Face pushes with a pre-configured model card.

## Important Decisions
1. **Focus on Quality over Quantity**: Decided to prune 3,845 records rather than risk training the LLM on ambiguous context mappings, ungrounded RagBench entries, or hallucinated formatting.
2. **Taxonomy Consolidation**: Explicitly scoped the current phase of FRED reproduction to Numerical and Temporal error types, matching the reality of the generated FinQA dataset.
3. **No Automatic API Generation**: Enforced a policy to halt data regeneration via Gemini until the entire validation and training strategy is bulletproof.

## Known Limitations
1. **Context Truncation**: The `format_docs` logic in `scripts/05_generate_cot_dataset.py` arbitrarily trims tables to 550 characters, breaking strict string-hashing against original FinQA sources.
2. **Zero Clean Samples**: The 1,155 valid rows consist *entirely* of synthetic errors. Without offline clean samples, the resulting model will suffer from over-editing bias (a 100% false-positive rate on clean inputs).
3. **Generic Financial Questions**: FinQA contains highly duplicate questions (e.g., "What is the net profit margin?"), requiring fuzzy-matching on the original target answer rather than just the question text to accurately disambiguate rows.

## Dataset Counts
- `synthetic_finqa_1500.jsonl`: 1,500 samples (Early prototype baseline; pruned from repo during cleanup)
- `synthetic_finqa_cot_5000.jsonl`: 5,000 samples (Unfiltered raw generation)
- `synthetic_finqa_cot_high_confidence_valid.jsonl`: 1,155 samples (Strictly validated, 447 canonical-aligned)
- `data/final/` Hybrid Dataset:
  - Total: 5,997 rows
  - Clean: 1,250 rows
  - Corrupted: 4,747 rows (447 Gemini, 4,300 Deterministic)
  - Train: 4,532 | Val: 624 | Test: 841

## Model and Training Decisions
1. **Sequence Length**: Token audit of ChatML CoT records revealed a maximum length of 1,083 tokens. Determined that setting `max_seq_length=2048` in Unsloth is perfectly safe, negating the need for the paper's highly expensive 8192 context window for this specific dataset.
2. **Training Parameters**: Adopted the paper's conservative baseline (`learning_rate=2e-5`, `num_train_epochs=2`) for the primary Kaggle experiment to prevent overfitting.
3. **Strict Split Design**: Planning an offline, deterministic Train/Val/Test split based on the RagBench `context` hash to ensure absolutely zero data leakage (document overlap) between sets.

## Next Steps
1. **Execute Baseline Training**: Run the prepared `notebooks/03_kaggle_unsloth_finetune.ipynb` notebook on Kaggle using a GPU environment, evaluate the final model, and push the adapter to Hugging Face.
2. **Perform Cleanup**: Execute the directory cleanup manifest to organize the workspace.

## Development & Pilot Pipeline History
> **Note on Pilot Scripts**: Transient experimentation scripts (`run_pilot.py`, `run_pilot_v2.py`, `validate_supplemental_dataset.py`, `audit_partial_supplemental.py`, `audit_official_finqa.py`) were used during the early exploratory phase to test API quotas, validity thresholds, and format fidelity. These were subsequently consolidated into the production numbered pipeline (`scripts/07_generate_supplemental_dataset.py` through `scripts/11_validate_hybrid_dataset.py`).

- `scripts/07_generate_supplemental_dataset.py`: A source-preserving supplemental dataset generator. It strictly pulls 1.0 adherence records from RagBench FinQA, avoids context summarization, implements configurable taxonomy targets, and outputs in structured JSON format while preserving `source_id` and raw documents.
- *(Historical Pilot)* `scripts/validate_supplemental_dataset.py`: Early XML/Span syntax validator designed for initial supplemental outputs (superseded by `11_validate_hybrid_dataset.py`).
- *(Historical Pilot)* `scripts/run_pilot.py` & `run_pilot_v2.py`: 50-sample pilot runners testing validity rates before scaling.

## Pilot Runs
- **Pilot V1**: Executed 50 requests using `gemini-3.5-flash-lite` on API Key 2. The script completed, but the validator returned a **20.0% valid rate** (10 valid, 40 invalid). The primary failure mode was `unrelated_text_changed_or_bad_replacement` because the model output only the XML tag instead of rewriting the full corrected text in the `Correction:` block.
- **Pilot V2**: Executed 50 requests using the updated prompt structure. The valid rate improved significantly to **83.7%**, successfully passing the 80% threshold required for bulk generation.
- **Bulk Generation**: Executed `scripts/08_run_bulk_generation.py` targeting 700 valid numerical samples and 500 valid temporal samples using API Keys 4 and 3 consecutively on `gemini-3.5-flash-lite`.
- **Bulk Generation Audit**: Interrupted at 90 samples to audit data quality. Evaluated matching against canonical FinQA math logic, informing the strict filtering in `09_audit_high_confidence_canonical.py` and final deterministic hybrid construction in `10_build_hybrid_dataset.py`.

## Base Model Evaluation with Paper Appendix E.2 Few-Shot Prompting
1. **Methodological Correction**: Identified that the zero-shot baseline scored 0.0 because it was prompted with unguided context without instructions or formatting definitions. Updated `scripts/12_run_evaluation.py` to implement the paper's exact Appendix E.2 few-shot prompt tailored to Numerical and Temporal error definitions, accompanied by a 1-shot in-context learning demonstration and JSON output specification.
2. **Parser Enhancement**: Updated `scripts/13_score_evaluation.py` with `extract_edited_text()` to parse JSON wrappers (`{"Edited": "..."}`) and markdown code blocks before evaluating XML tag validity.
3. **Execution & Checkpointing**: Ran `qwen2.5:7b` through the updated few-shot prompt. To ensure hardware safety on laptop GPU, execution was paused at record 435 with full checkpointing preserved in `data/eval/predictions_base.jsonl`.
4. **Empirical Findings (435 Samples)**:
   - **Binary Detection**: Base model achieved 82.2 F1 (vs. FRED's 91.8 on the matched slice).
   - **Numerical Errors**: Base model achieved 76.7 F1 (vs. FRED's 85.5).
   - **Temporal Errors**: Base model struggled severely with temporal attribution, scoring only 15.9 F1 (vs. FRED's 66.0).
   - **Exact Editing (CES)**: In-context few-shot learning collapsed on precise span deletion and replacement, yielding only **7.9% Complete Edit Success** (CES) and 55.0% XML format validity.
   - **Fine-Tuning Advantage**: Fine-tuned `fred:latest` demonstrated a **nearly 6× improvement in exact editing (46.9% CES)** and near-perfect XML syntax compliance (97.0%), proving that domain-specific fine-tuning is required for reliable in-place hallucination correction.
