# NVFP4 Long-Context Debug Status Report

## Executive Summary

This debug session started from a clear symptom: the quantized NVFP4 Qwen3-235B-A22B model produced normal RULER answers at short context, then gradually degraded into repeated phrases and repeated single words as context length increased.

The current best conclusion is:

- the issue is **not** just decoding
- the issue is **not** YaRN alone
- the issue is **not** NVFP4 quantization alone inside the native context range
- the most likely failing condition is **YaRN + NVFP4 quantized inference path**

More precisely, the problem appears when long-context extension is enabled together with the NVFP4 path. Later follow-up tests narrowed the issue substantially, and newer 16k/32k artifacts in this repo are consistent with that narrower conclusion.

## Scope of This Report

This report combines:

- evidence verified directly from files in this repo
- findings already written in `docs/findings/2026-05-08-longctx-repetition-analysis.md`
- follow-up conclusions reported in the debug session on 2026-05-09

Important caveat:

- the newer result files do not record the full serve flags, so the exact “with YaRN” vs “without YaRN” setting is partly inferred from the session notes rather than fully reconstructable from result JSON alone

## Problem Statement

The target problem was a long-context accuracy issue on the RULER dataset for the NVFP4-quantized Qwen3-235B-A22B model.

The main symptom was:

- short-context outputs looked normal
- longer-context outputs started repeating numbers or short phrases
- very long-context outputs collapsed into low-diversity loops such as `and, and, and`

## Main Artifacts

- Response extraction script: `parse_ruler_responses.py`
- Merged analysis file: `analysis/ruler_first4_responses.json`
- Earlier deep-dive findings note: `docs/findings/2026-05-08-longctx-repetition-analysis.md`

## Current Test Summary

### File-Verified Runs

| Test | Condition | Result | Evidence | Note |
|---|---|---|---|---|
| Short-context baseline | NVFP4, `seq_4096` | `1.0`, normal answer | `results_2026-05-08T11-52-23.136565.json`, `analysis/ruler_first4_responses.json:22` | Healthy baseline |
| First degradation point | NVFP4, `seq_8192` | `0.8125`, repeated numbers start | `results_2026-05-08T11-42-20.213252.json:7`, `analysis/ruler_first4_responses.json:83` | Still partly correct |
| Initial bad 16k run | NVFP4, `seq_16384` | `0.0`, phrase repetition | `results_2026-05-08T11-49-27.935194.json:7`, `analysis/ruler_first4_responses.json:148` | Long-context failure visible |
| Initial bad 32k run | NVFP4, `seq_32768` | `0.0`, `and, and, and` loop | `results_2026-05-08T11-38-06.003485.json:7`, `analysis/ruler_first4_responses.json:213` | Severe collapse |
| Initial bad 128k run | NVFP4, `seq_131072` | `0.0`, `the time`-style loop | `results_2026-05-08T10-57-50.542877.json:7`, `analysis/ruler_first4_responses.json:278` | Severe collapse |
| Later good 16k run | NVFP4, `seq_16384` | `1.0`, normal answer | `results_2026-05-08T12-53-34.618507.json:7`, `samples_niah_multiquery_2026-05-08T12-53-34.618507.jsonl` | Remove Yarn |
| Later good 32k run | NVFP4, `seq_32768` | `1.0`, normal answer | `results_2026-05-08T12-59-45.943858.json:7`, `samples_niah_multiquery_2026-05-08T12-59-45.943858.jsonl` | Remove Yarn |

### Session-Reported Findings

| Finding | Status | Interpretation |
|---|---|---|
| Changing temperature | Did not fix the issue | The problem is not explained by temperature alone |
| `BF16 + YaRN` | Good | YaRN alone is unlikely to be the primary root cause |
| `NVFP4 + no YaRN` at `16k/32k` | Good | NVFP4 is fine in the tested native-range setup |
| NVFP4-only failure | Confirmed | The interesting failure is on the NVFP4 path, not a general model failure |
| Current best hypothesis | `YaRN + NVFP4` interaction | Most likely failing condition based on the current evidence |

## Timeline

### 1. Initial Symptom Capture

We created a parser to extract the first 4 question/response pairs from each RULER run and merged them into one analysis file.

Relevant artifacts:

- `parse_ruler_responses.py`
- `analysis/ruler_first4_responses.json`

### 2. First Verified Failure Snapshot

From the merged file and per-run result JSONs:

- `seq_4096` was healthy and scored `1.0`
- `seq_8192` still partially worked and scored `0.8125`
- `seq_16384`, `seq_32768`, and `seq_131072` collapsed and scored `0.0`

Evidence:

- `analysis/ruler_first4_responses.json:22`
- `analysis/ruler_first4_responses.json:83`
- `analysis/ruler_first4_responses.json:148`
- `analysis/ruler_first4_responses.json:213`
- `analysis/ruler_first4_responses.json:278`
- `quantized_model_nvfp4-tp4-eval/seq_4096/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-52-23.136565.json:5-8`
- `quantized_model_nvfp4-tp4-eval/seq_8192/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-42-20.213252.json:5-8`
- `quantized_model_nvfp4-tp4-eval/seq_16384/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-49-27.935194.json:5-8`
- `quantized_model_nvfp4-tp4-eval/seq_32768/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-38-06.003485.json:5-8`
- `quantized_model_nvfp4-tp4-eval/seq_131072/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T10-57-50.542877.json:5-8`

Qualitative pattern:

- `seq_8192`: repeated the right numbers
- `seq_16384`: repeated short phrases like `the same way`
- `seq_32768`: collapsed into `and, and, and`
- `seq_131072`: collapsed into loops like `the time the time`

### 3. First Hypothesis Set

The first round of hypotheses was:

- greedy decoding may be amplifying repetition
- the YaRN long-context extension path may be involved
- the vLLM runtime path may be involved
- pure attention-weight quantization is less likely, because attention projections are excluded from quantization in the checkpoint config

Relevant evidence:

- `run_evaluation.sh:225-243`
- `quantized_model_nvfp4-tp4-eval/vllm_server.log:7`
- `/media/jenkins/saved_models/Qwen3-235B-A22B_nvfp4_LLMC/quantized_model_nvfp4/config.json:16`
- `/media/jenkins/saved_models/Qwen3-235B-A22B_nvfp4_LLMC/quantized_model_nvfp4/config.json:68-71`
- `/media/jenkins/saved_models/Qwen3-235B-A22B_nvfp4_LLMC/quantized_model_nvfp4/config.json:549`

### 4. Follow-Up Narrowing

The later follow-up tests changed the picture substantially.

Session-reported findings:

- changing temperature did **not** fix the issue
- `BF16 + YaRN` was good
- `NVFP4 + no YaRN` was good at `16k/32k`
- the user confirmed the issue is **NVFP4-only**

Interpretation:

- this weakens the “greedy decoding is the main root cause” theory
- this weakens the “YaRN by itself is broken” theory
- this weakens the “NVFP4 is always broken” theory
- this strengthens the interaction hypothesis: **YaRN + NVFP4**

### 5. Newer On-Disk Results Are Consistent With the Narrower Conclusion

There are newer 16k/32k result files in `quantized_model_nvfp4-tp4-eval/` that look healthy:

- `seq_16384` later run scored `1.0`: `quantized_model_nvfp4-tp4-eval/seq_16384/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T12-53-34.618507.json:7`
- `seq_32768` later run scored `1.0`: `quantized_model_nvfp4-tp4-eval/seq_32768/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T12-59-45.943858.json:7`

The corresponding sample outputs look normal:

- `quantized_model_nvfp4-tp4-eval/seq_16384/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/samples_niah_multiquery_2026-05-08T12-53-34.618507.jsonl`
- `quantized_model_nvfp4-tp4-eval/seq_32768/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/samples_niah_multiquery_2026-05-08T12-59-45.943858.jsonl`

This matters because the same repo also contains an earlier `seq_32768` failing run from the same day:

- bad `32768` run: `quantized_model_nvfp4-tp4-eval/seq_32768/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T12-42-28.373331.json:7`
- good `32768` run: `quantized_model_nvfp4-tp4-eval/seq_32768/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T12-59-45.943858.json:7`

That supports the idea that a specific run condition changed, rather than the model being uniformly broken.

## Current Best Conclusion

The current best conclusion is:

**The root cause is most likely the interaction between YaRN and the NVFP4 quantized inference path.**

That does **not** yet prove whether the bug sits in:

- the quantized checkpoint itself
- activation quantization behavior under YaRN-scaled positions
- the compressed-tensors runtime path
- the vLLM long-context execution path
- or some combination of the above

But it is already much narrower than the original problem statement.

## What Is Likely Ruled Out

Based on the current state, these are less likely as the primary root cause:

- **pure temperature / sampling issue**
  - because changing temperature did not fix it
- **YaRN alone**
  - because `BF16 + YaRN` was reported good
- **NVFP4 alone at native context**
  - because `NVFP4 + no YaRN` was reported good at `16k/32k`
- **attention projection quantization as the first suspect**
  - because `self_attn.*` and `lm_head` are excluded in the quantization ignore list

## What Is Still Unknown

The main unanswered question is:

**Is this a model-layer problem or a runtime-path problem?**

That distinction matters:

- if the failure reproduces in a direct forward pass, then it may be possible to identify the first bad layer
- if the failure appears only inside vLLM serving, then the failure may live in the cache / rope / kernel / scheduling path rather than in a single transformer layer

## Best Next Debug Question

The best next question is:

**Can the failure be reproduced outside the serving stack?**

If yes:

- compare BF16 and NVFP4 on the exact same prompt under teacher forcing
- record hidden-state similarity layer by layer
- inspect the first layer where the last-token hidden state diverges sharply
- then split that layer into:
  - input to layer
  - post-attention residual
  - post-MLP or post-expert residual

If no:

- the bug is more likely in the vLLM runtime path
- then the next work should target:
  - YaRN application
  - KV/cache path
  - chunked prefill
  - graph/compile path
  - compressed-tensors execution

## Layer-Isolation Guidance

If you want to isolate the first bad layer later, the clean procedure is:

1. Pick one prompt that fails reliably under the known-bad condition.
2. Feed the exact same tokens to BF16 and NVFP4.
3. Do **not** let generation branch early.
4. Compare the last prompt position hidden state after every layer.
5. Look for the first large jump in cosine distance or relative error.
6. Inside that layer, compare:
   - layernorm output
   - attention block output
   - MLP or expert block output
   - residual output

Because the current checkpoint excludes `self_attn.*` from quantization, the first place I would inspect is:

- MLP or expert projections
- activation quantization behavior
- expert routing behavior after YaRN-scaled long-context inputs

## Practical Blog Angle

This debug story is already strong enough for a blog post because it has a clean arc:

1. **Symptom**
   - long-context answers collapse into repetitions
2. **Initial suspicion**
   - maybe decode settings, maybe YaRN, maybe quantization
3. **Evidence gathering**
   - extract a small, readable slice of responses instead of inspecting giant files manually
4. **Narrowing tests**
   - BF16 + YaRN good
   - NVFP4 + no YaRN good
   - therefore the interesting failure is the interaction
5. **Current state**
   - the issue is narrowed, but the exact failing layer or runtime component is still open

## Suggested Blog Outline

### Title ideas

- `Debugging an NVFP4 Long-Context Failure in Qwen3: When YaRN and Quantization Interact`
- `From "and, and, and" to a Root-Cause Hypothesis: Debugging Long-Context Collapse in NVFP4`
- `Why a Quantized Model Failed at Long Context: A RULER Debugging Story`

### Section outline

1. Background
   - what model
   - what task
   - why long context mattered
2. Symptom
   - show one clean short-context answer and one broken long-context answer
3. First pass analysis
   - merged parser
   - response slices
   - metric cliff
4. Competing hypotheses
   - decoding
   - YaRN
   - quantization
   - runtime
5. Key ablations
   - BF16 + YaRN
   - NVFP4 + no YaRN
   - why that changed the conclusion
6. Current root-cause hypothesis
   - YaRN × NVFP4 interaction
7. What remains open
   - first bad layer vs runtime bug
8. Next steps
   - direct forward compare
   - runtime ablations

## Recommended Supporting Figures for the Blog

- a 5-row table with `4096 / 8192 / 16384 / 32768 / 131072`
- one short “good output vs bad output” comparison
- one timeline figure:
  - initial failure
  - decode test
  - BF16 + YaRN
  - NVFP4 without YaRN
  - current conclusion

## Bottom Line

At the current stage, the debugging work has already moved from “the model fails on long context” to a much tighter statement:

**the failure appears when YaRN is combined with the NVFP4 path, while BF16 + YaRN and NVFP4 without YaRN remain good in the tested ranges.**

That is strong enough to write about now, even before the exact failing layer or runtime component is fully isolated.
