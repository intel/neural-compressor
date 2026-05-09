# Long-Context Repetition Debug Plan

## Question

Why do `filtered_response` outputs degrade into repeated phrases and then repeated single words as `seq_length` increases in the RULER runs, and what should be checked next?

## Sources to Cross-Reference

- Consolidated extraction output in `analysis/ruler_first4_responses.json`
- Raw `samples_niah_multiquery_*.jsonl` rows from selected `seq_*` runs
- Evaluation launch script and model-serving settings in `run_evaluation.sh`
- Quantized model config files under `qmodels_nvfp4/quantized_model_nvfp4/`
- Any relevant runtime logs under `quantized_model_nvfp4-tp4-eval/`

## Finishing Criteria

- Confirm the repetition pattern from targeted samples only, without dumping the full merged file.
- Identify at least 2 source categories that support the likely cause analysis.
- Save a findings note with evidence-backed hypotheses and concrete debug steps.
