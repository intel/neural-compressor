# Long-Context Repetition in RULER Outputs

## Question

Why do the `filtered_response` values collapse into repeated phrases and then repeated single words as `seq_length` increases, and what should be debugged next?

## Methodology

- Checked targeted `filtered_response` slices from the merged file instead of dumping the full file.
- Cross-checked against raw `samples_niah_multiquery_*.jsonl` rows and per-run `results_*.json`.
- Reviewed the RULER serve/eval path in `run_evaluation.sh`.
- Reviewed the actual external checkpoint config used by the run.
- Compared findings against the official Qwen model card guidance for long-context YaRN and decoding.

## Findings

1. **The degeneration pattern is real and grows with context length.**

   Evidence from the merged extraction:

   - `seq_4096` answers are normal and task-relevant: `analysis/ruler_first4_responses.json:22`, `:35`, `:48`, `:61`
   - `seq_8192` starts repeating the extracted numbers: `analysis/ruler_first4_responses.json:83`, `:97`, `:111`, `:125`
   - `seq_16384` collapses into short repeated phrases like `the same way`: `analysis/ruler_first4_responses.json:148`, `:162`, `:176`, `:190`
   - `seq_32768` collapses almost entirely into `and, and, and`: `analysis/ruler_first4_responses.json:213`, `:227`, `:241`, `:255`
   - `seq_131072` collapses into loops like `the time the time` and even long repeated character tails: `analysis/ruler_first4_responses.json:278`, `:292`, `:306`, `:320`

   The metric cliff matches that qualitative change:

   - `seq_8192` still scores `0.8125`: `quantized_model_nvfp4-tp4-eval/seq_8192/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-42-20.213252.json:7`
   - `seq_16384`, `seq_32768`, and `seq_131072` all score `0.0`: `quantized_model_nvfp4-tp4-eval/seq_16384/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-49-27.935194.json:7`, `quantized_model_nvfp4-tp4-eval/seq_32768/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-38-06.003485.json:7`, `quantized_model_nvfp4-tp4-eval/seq_131072/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T10-57-50.542877.json:7`

   I also measured token diversity across the first 4 responses of each run:

   - `seq_4096`: top tokens are still task words such as `numbers`, `are`
   - `seq_32768`: `and,` appears 248 times across only 4 short outputs
   - `seq_131072`: `the` appears 133 times and `time` 54 times

   Command used:

   ```bash
   python3 - <<'PY'
   import json
   from pathlib import Path
   from collections import Counter
   for seq in [4096,8192,16384,32768,131072]:
       src = sorted([p for p in Path(f'quantized_model_nvfp4-tp4-eval/seq_{seq}').rglob('samples_niah_multiquery_*.jsonl') if 'copy' not in p.name.lower()])[0]
       with src.open() as f:
           texts = [json.loads(next(f))['filtered_resps'][0] for _ in range(4)]
       tokens = ' '.join(texts).split()
       print(seq, Counter(tokens).most_common(5))
   PY
   ```

2. **The eval is using greedy decoding, and Qwen explicitly warns that greedy decoding can cause endless repetitions.**

   Local evidence:

   - The actual eval config is `do_sample: false`, `temperature: 0.0`, `max_gen_toks: 128`: `quantized_model_nvfp4-tp4-eval/seq_8192/__media__jenkins__saved_models__Qwen3-235B-A22B_nvfp4_LLMC__quantized_model_nvfp4/results_2026-05-08T11-42-20.213252.json:78-82`

   External primary-source evidence:

   - The official Qwen model card warns: “we do not recommend using greedy decoding, which can lead to endless repetitions.” It also recommends sampling-based decoding defaults for Qwen3. Source: official model card `Qwen/Qwen3-235B-A22B-MLX-4bit`

   Interpretation:

   - This is a strong suspect. The outputs are not random noise; they are classic greedy-loop failures that get worse as the prompt gets longer and the model becomes less certain.

3. **The serve path forces static long-context YaRN even though the checkpoint metadata itself does not carry long-context rope settings.**

   Serve-path evidence:

   - `run_evaluation.sh` hard-codes:
     - `ROPE_SCALING_JSON='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`: `run_evaluation.sh:225`
     - `ROPE_VALUE="{\"rope_scaling\":${ROPE_SCALING_JSON},\"max_position_embeddings\":131072}"`: `run_evaluation.sh:228`
     - these overrides are passed into `vllm serve`: `run_evaluation.sh:235-243`

   Runtime evidence:

   - The actual server started with:
     - `max_model_len: 131072`
     - `hf_overrides: {'rope_scaling': {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}, 'max_position_embeddings': 131072}`
     - `quantization=compressed-tensors`
     - `enable_prefix_caching=True`
     - `enable_chunked_prefill=True`
     - `enforce_eager=False`
     - vLLM version `0.20.2rc1.dev92+g20cac26b1`
     - See `quantized_model_nvfp4-tp4-eval/vllm_server.log:7`, `:15`, `:19`, `:25`

   Checkpoint evidence:

   - The actual checkpoint config says:
     - `max_position_embeddings: 40960`: `/media/jenkins/saved_models/Qwen3-235B-A22B_nvfp4_LLMC/quantized_model_nvfp4/config.json:16`
     - `rope_scaling: null`: `/media/jenkins/saved_models/Qwen3-235B-A22B_nvfp4_LLMC/quantized_model_nvfp4/config.json:549`

   Interpretation:

   - This does not prove the YaRN override is wrong, but it does prove the long-context behavior depends on runtime overrides rather than checkpoint-native metadata.
   - That makes the long-context extension path a first-class debug target.

4. **Attention projections are excluded from quantization, so “attention weights got quantized and broke long context” is not the best first explanation.**

   Evidence from the checkpoint config:

   - Inputs and weights for targeted linear layers are 4-bit: `/media/jenkins/saved_models/Qwen3-235B-A22B_nvfp4_LLMC/quantized_model_nvfp4/config.json:32-39`
   - But the ignore list includes `re:.*self_attn.*` and `lm_head`: `/media/jenkins/saved_models/Qwen3-235B-A22B_nvfp4_LLMC/quantized_model_nvfp4/config.json:68-71`

   Interpretation:

   - Direct attention-projection quantization is less likely to be the main cause here.
   - Quantized MLP / expert paths and activation quantization can still amplify long-context instability, so quantization remains a plausible secondary cause.

5. **The runtime stack is aggressive enough that a correctness bug is still on the table.**

   Evidence:

   - The server is a dev build: `quantized_model_nvfp4-tp4-eval/vllm_server.log:3`
   - Chunked prefill is enabled: `quantized_model_nvfp4-tp4-eval/vllm_server.log:19`
   - Prefix caching is enabled and eager mode is off: `quantized_model_nvfp4-tp4-eval/vllm_server.log:25`

   Interpretation:

   - If this is a compressed-tensors + long-context + chunked-prefill/cudagraph interaction, the symptom can look exactly like low-diversity loops rather than a crash.

## Most Likely Causes

1. **Greedy decoding with Qwen3**
   - Highest-confidence because the official Qwen guidance warns about this exact failure mode, and your eval is doing greedy decode.

2. **Long-context YaRN extension path**
   - High-confidence because long context is not checkpoint-native in the saved config; it is being imposed at serve time.
   - If the extension recipe is slightly off for this checkpoint/runtime, errors should grow with sequence length exactly as observed.

3. **Runtime optimization bug in this vLLM + compressed-tensors path**
   - Medium-confidence because you are on a dev vLLM build with chunked prefill, prefix caching, cudagraphs, and quantized custom kernels enabled.

4. **Quantized MLP / activation path amplifying instability**
   - Medium-confidence as an amplifier, lower-confidence as the only root cause.

## Recommended Debug Order

1. **Remove greedy decoding first.**

   Run one bad prompt, preferably `seq_32768`, with Qwen’s recommended sampling-style decode instead of:

   - current: `do_sample=False`, `temperature=0.0`
   - try: model-card defaults from `generation_config.json` or the official Qwen recommendation

   If the repetition disappears immediately, you have found at least one major contributor.

2. **Compare one prompt on the original non-quantized model with the same serve path.**

   Keep everything else fixed:

   - same prompt
   - same vLLM version
   - same YaRN override
   - same decode settings

   Outcomes:

   - if BF16 also repeats, the issue is probably decode or runtime
   - if BF16 is fine and NVFP4 repeats, the issue is probably checkpoint or quantized runtime

3. **Ablate the long-context override.**

   For a `seq_32768` prompt:

   - run once without YaRN
   - run once with the current YaRN override

   Because the checkpoint advertises `40960` native positions, `32768` should be a clean place to test whether YaRN itself is hurting quality.

4. **Ablate runtime optimizations.**

   Test one failing prompt with safer execution:

   - eager mode on
   - prefix caching off
   - chunked prefill off
   - if possible, fewer graph/compile optimizations

   This isolates correctness bugs in the serving stack.

5. **Compare vLLM against a plain Transformers/HF generate path on one prompt.**

   You do not need a full benchmark.

   One prompt at `32768` is enough to answer:

   - model/checkpoint issue
   - or serving/runtime issue

6. **Inspect logits around the first repetition loop.**

   The key question is whether:

   - EOS disappears from the top candidates
   - one token becomes pathologically dominant
   - or a tiny set of tokens (`and`, `the`, punctuation) collapses the distribution

   That distinguishes “decode problem” from “model state is already corrupted before decode.”

## Practical Next Experiment

If time is limited, the fastest high-value experiment is:

1. pick the first `seq_32768` prompt
2. run it once with current settings
3. run it once with non-greedy Qwen-recommended decode
4. run it once with non-greedy decode plus eager mode / no chunked prefill
5. compare outputs side by side

That single 3-run ablation should tell you whether the first problem is decoding, runtime, or the checkpoint itself.

## Open Questions

- I did not re-run the 235B model live in this session, so the ranking above is evidence-based but still inferential.
- I did not have the original pre-quantized Qwen3-235B-A22B checkpoint config locally, only the quantized exports and the official Qwen model card guidance.
