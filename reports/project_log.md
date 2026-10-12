# Project Log

## Completed Work
1. **Repository Audit**: Performed non-destructive audits of all generator scripts, grounding metadata, token lengths, and XML syntax.
2. **Dataset Diagnosis**: Evaluated the 5,000-row synthetic CoT FinQA dataset (`synthetic_finqa_cot_5000.jsonl`).
3. **Strict Validation Pipeline**: Engineered a highly precise validation and fuzzy-matching pipeline to discard poorly generated reasoning, malformed XML, and heavily modified facts.
4. **Dataset Scrubbing**: Generated `synthetic_finqa_cot_high_confidence_valid.jsonl`, isolating 1,155 strictly grounded, structurally sound, and fully verifiable records from the initial 5,000 generated samples.
5. **Final Hybrid Dataset Build**: Built `scripts/10_build_hybrid_dataset.py` to deterministically create 5,997 structured Train/Val/Test rows, combining 447 fully canonical-aligned LLM seed records, deterministic Python numerical and temporal corruptions, and 1,250 clean uncorrupted records.
6. **Final Dataset Validation**: Built `scripts/11_validate_hybrid_dataset.py` applying 13 strict checks (source leakage, XML validity, span reconstruction) confirming a 100% pass rate.
7. **Training Notebook Preparation**: Refactored `notebooks/03_kaggle_unsloth_finetune_cot.ipynb` to load the 5,997-row ChatML dataset, configure a Qwen2.5-7B-Instruct 4-bit QLoRA run with response-only loss masking, enforce strict XML formats (without free-form CoT), run smoke tests, and explicitly handle Hugging Face pushes with a pre-configured model card.

## Important Decisions
1. **Focus on Quality over Quantity**: Decided to prune 3,845 records rather than risk training the LLM on ambiguous context mappings, ungrounded RagBench entries, or hallucinated formatting.
2. **Taxonomy Consolidation**: Explicitly scoped the current phase of FRED reproduction to Numerical and Temporal error types, matching the reality of the generated FinQA dataset.
3. **No Automatic API Generation**: Enforced a policy to halt data regeneration via Gemini until the entire validation and training strategy is bulletproof.

## Known Limitations
1. **Context Truncation**: The `format_docs` logic in `scripts/05_generate_cot_dataset.py` arbitrarily trims tables to 550 characters, breaking strict string-hashing against original FinQA sources.
2. **Zero Clean Samples**: The 1,155 valid rows consist *entirely* of synthetic errors. Without offline clean samples, the resulting model will suffer from over-editing bias (a 100% false-positive rate on clean inputs).
3. **Generic Financial Questions**: FinQA contains highly duplicate questions (e.g., "What is the net profit margin?"), requiring fuzzy-matching on the original target answer rather than just the question text to accurately disambiguate rows.

## Dataset Counts
- `synthetic_finqa_1500.jsonl`: 1,500 samples (Baseline)
- `synthetic_finqa_cot_5000.jsonl`: 5,000 samples (Unfiltered)
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
1. **Execute Baseline Training**: Run the prepared `03_kaggle_unsloth_finetune_cot.ipynb` notebook on Kaggle using a GPU environment, evaluate the final model, and push the adapter to Hugging Face.
2. **Perform Cleanup**: Execute the directory cleanup manifest to organize the workspace.

## Newly Implemented Code
- `scripts/07_generate_supplemental_dataset.py`: A source-preserving supplemental dataset generator. It strictly pulls 1.0 adherence records from RagBench FinQA, avoids context summarization, implements configurable taxonomy targets, and outputs in structured JSON format while preserving `source_id` and raw documents.
- `scripts/validate_supplemental_dataset.py`: A highly rigorous XML/Span syntax validator designed exclusively for the new supplemental outputs.
- `scripts/run_pilot.py`: An isolated script running a 50-sample execution of the new pipeline to test validity rates (target >= 80%).
- `scripts/run_pilot_v2.py`: Updated isolated script running a 50-sample execution to fix the shortcutting issue in pilot V1.

## Pilot Runs
- **Pilot V1**: Executed 50 requests using `gemini-3.5-flash-lite` on API Key 2. The script completed, but the validator returned a **20.0% valid rate** (10 valid, 40 invalid). The primary failure mode was `unrelated_text_changed_or_bad_replacement` because the model output only the XML tag instead of rewriting the full corrected text in the `Correction:` block.
- **Pilot V2**: Executed 50 requests using the updated prompt structure. The valid rate improved significantly to **83.7%**, successfully passing the 80% threshold required for bulk generation.
- **Bulk Generation**: Currently executing `scripts/08_run_bulk_generation.py` targeting 700 valid numerical samples and 500 valid temporal samples. The script utilizes API Keys 4 and 3 consecutively (handling `429` Quota Exhausted limits) on `gemini-3.5-flash-lite`.
- **Bulk Generation Interruption**: The bulk generation was intentionally interrupted at 90 samples to audit data quality. The partial output was preserved and audited via `scripts/audit_partial_supplemental.py`. See `reports/bulk_generation_interruption_report.md` for full breakdown.
- **Official FinQA Audit**: The interrupted generated records were carefully matched against the canonical Official FinQA dataset (`train.json`, `dev.json`, `test.json`) using `scripts/audit_official_finqa.py`. The audit checked for exact one-edit differences, canonical alignment, and correct math logic. We found 57 multiple-edit failures and 60 source canonical-mismatch failures, indicating that while structurally valid XML was generated, the semantic logic and edit consistency need further refinement before retraining. Results saved to `reports/official_finqa_audit.md` and `data/audit/official_finqa_audit.jsonl`.
