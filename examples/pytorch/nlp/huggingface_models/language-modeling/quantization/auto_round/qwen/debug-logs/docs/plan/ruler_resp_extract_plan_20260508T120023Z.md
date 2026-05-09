# RULER Response Extraction Plan

## Goal

Add a Python script that scans the existing `seq_*` evaluation outputs, extracts the model response for each question from the `samples_niah_multiquery_*.jsonl` files, keeps only the first 4 questions per run, and writes one new consolidated file for later long-context analysis.

## Observed Input Shape

- Data lives under `quantized_model_nvfp4-tp4-eval/seq_*`.
- Each run directory contains one primary `samples_niah_multiquery_*.jsonl` file plus some `copy`/helper artifacts that should be ignored.
- Each JSONL row contains:
  - `doc_id`
  - `doc.input`
  - `resps`
  - `filtered_resps`
  - per-metric fields such as `4096` or `131072`

## Implementation Choices

- Auto-discover `seq_*` directories instead of hardcoding lengths.
- Prefer the canonical `.jsonl` file and skip filenames containing `copy`.
- Read only the first 4 rows from each discovered run.
- Store both the extracted question text and the chosen response text so later analysis does not have to reopen the original large prompt file.
- Write JSON output because it is easy to diff, inspect, and load in later analysis steps.

## Output Shape

Planned output file:

- `analysis/ruler_first4_responses.json`

Each run entry should include:

- `seq_length`
- source file path
- up to 4 extracted records
- per record:
  - `question_index`
  - `doc_id`
  - parsed question text
  - extracted response text
  - optional metric snapshot from the row

## Finishing Criteria

- Script runs from this repo without manual path edits.
- Script discovers the current `seq_4096`, `seq_8192`, `seq_16384`, `seq_32768`, and `seq_131072` runs automatically.
- Output file is created successfully.
- Output includes no more than 4 records per run.
- At least one generated record is checked against the source JSONL to confirm the extracted question and response match the raw data.
