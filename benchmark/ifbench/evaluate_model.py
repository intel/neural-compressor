#!/usr/bin/env python3
"""End-to-end IFBench evaluation of the model configured in ``.env``.

Mirrors the NVIDIA nemotron-3-ultra ``nemo_skills.ns_ifbench`` recipe: it takes a
single *policy model* (the OpenAI-compatible API defined in ``.env``), generates
responses for the IFBench test set, then scores instruction-following with the
local rule-based checkers in ``evaluation_lib`` (both *strict* and *loose*).

Key differences from the two-step ``generate_responses.py`` + ``run_eval.py`` flow:

* One command does generation + evaluation.
* Supports ``--num-repeats`` consensus (nemotron uses 8) and reports the mean.
* Responses are aligned to inputs **by key**, so the prompt-text mismatch problem
  (extra whitespace / scrambled rows) can never happen.

Usage::

    /data/miniforge3/bin/python3 evaluate_model.py                 # uses .env
    /data/miniforge3/bin/python3 evaluate_model.py --num-repeats 8
    /data/miniforge3/bin/python3 evaluate_model.py --limit 10      # smoke test
"""

import argparse
import collections
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import evaluation_lib
import httpx
from config import get_settings
from tqdm import tqdm


def generate_response(
    client,
    api_base,
    model,
    prompt,
    temperature,
    top_p,
    max_tokens,
    enable_thinking,
    api_key,
    seed,
    request_timeout,
    max_retries,
):
    """Call an OpenAI-compatible /chat/completions endpoint for one prompt.

    Aligned with the Nemotron 3 Ultra recipe: send temperature/top_p, omit
    max_tokens when <=0 (server default, avoids truncating long thinking output),
    enable thinking via chat_template_kwargs, use a long request_timeout and retry
    with backoff (Nemotron: request_timeout 3600, max_retries 10).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if seed is not None:
        payload["seed"] = seed

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=request_timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(min(2**attempt, 30))
            continue
    raise last_exc


def generate_all(inputs, args, seed):
    """Generate one response per input, keyed by input.key.

    Errors -> ''.
    """
    responses = {}
    errors = []
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_inp = {
                executor.submit(
                    generate_response,
                    client,
                    args.api_base,
                    args.model,
                    inp.prompt,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.enable_thinking,
                    args.api_key,
                    seed,
                    args.request_timeout,
                    args.max_retries,
                ): inp
                for inp in inputs
            }
            with tqdm(total=len(inputs), desc=f"Generating (seed={seed})") as pbar:
                for future in as_completed(future_to_inp):
                    inp = future_to_inp[future]
                    try:
                        responses[inp.key] = future.result()
                    except Exception as e:
                        responses[inp.key] = ""
                        errors.append((inp.key, str(e)))
                    pbar.update(1)
    if errors:
        print(f"  {len(errors)} generation error(s); first few:")
        for k, msg in errors[:5]:
            print(f"    - key {k}: {msg}")
    return responses


def score(inputs, responses):
    """Return (strict_outputs, loose_outputs) for one set of responses.

    Responses are aligned by key -> we build the prompt->response dict the
    evaluation library expects, guaranteeing an exact match for every input.
    """
    prompt_to_response = {inp.prompt: responses[inp.key] for inp in inputs}
    strict = [evaluation_lib.test_instruction_following_strict(i, prompt_to_response) for i in inputs]
    loose = [evaluation_lib.test_instruction_following_loose(i, prompt_to_response) for i in inputs]
    return strict, loose


def summarize(outputs):
    """Compute prompt-level & instruction-level accuracy and per-type breakdown."""
    prompt_total = prompt_correct = 0
    instr_total = instr_correct = 0
    per_type_total = collections.defaultdict(int)
    per_type_correct = collections.defaultdict(int)
    for o in outputs:
        prompt_total += 1
        if all(o.follow_instruction_list):
            prompt_correct += 1
        instr_total += len(o.instruction_id_list)
        instr_correct += sum(o.follow_instruction_list)
        for iid, ok in zip(o.instruction_id_list, o.follow_instruction_list):
            per_type_total[iid] += 1
            per_type_correct[iid] += int(ok)
    return {
        "prompt_level": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_level": instr_correct / instr_total if instr_total else 0.0,
        "per_type": {iid: per_type_correct[iid] / per_type_total[iid] for iid in sorted(per_type_total)},
    }


def main():
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="End-to-end IFBench evaluation of the .env model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-base", default=settings.api_base)
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--api-key", default=settings.api_key)
    parser.add_argument("--input-file", default=settings.input_file)
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument("--temperature", type=float, default=settings.temperature)
    parser.add_argument("--top-p", type=float, default=settings.top_p)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.max_tokens,
        help="<=0 omits max_tokens (server default, Nemotron-style)",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=settings.enable_thinking,
        help="Send chat_template_kwargs.enable_thinking=true",
    )
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false", help="Disable thinking mode")
    parser.add_argument("--seed", type=int, default=settings.seed, help="Base seed; repeat i uses seed+i")
    parser.add_argument("--workers", type=int, default=settings.workers)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=settings.request_timeout,
        help="Per-request timeout in seconds (Nemotron: 3600)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=settings.max_retries,
        help="Retries per request on timeout/error (Nemotron: 10)",
    )
    parser.add_argument(
        "--num-repeats", type=int, default=settings.num_repeats, help="Consensus repeats (Nemotron uses 8)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N prompts (smoke test)")
    parser.add_argument("--save-responses", action="store_true", help="Also dump generated responses per repeat")
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (or set MODEL in .env)")
    if not args.api_base:
        parser.error("--api-base is required (or set API_BASE in .env)")

    inputs = evaluation_lib.read_prompt_list(args.input_file)
    if args.limit:
        inputs = inputs[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "-")

    print("=" * 64)
    print(f"Model:        {args.model}")
    print(f"API:          {args.api_base}")
    print(f"Test set:     {args.input_file} ({len(inputs)} prompts)")
    print(f"num_repeats:  {args.num_repeats}  temperature={args.temperature}" f"  top_p={args.top_p}")
    mt = args.max_tokens if args.max_tokens and args.max_tokens > 0 else "server-default"
    print(f"max_tokens:   {mt}  enable_thinking={args.enable_thinking}")
    print(f"timeout:      {args.request_timeout}s  max_retries={args.max_retries}" f"  workers={args.workers}")
    print("=" * 64)

    repeat_summaries = {"strict": [], "loose": []}

    for r in range(args.num_repeats):
        seed = None if args.seed is None else args.seed + r
        print(f"\n[repeat {r + 1}/{args.num_repeats}]")
        responses = generate_all(inputs, args, seed)

        if args.save_responses:
            resp_path = out_dir / f"{safe_model}-responses-r{r}.jsonl"
            with open(resp_path, "w") as f:
                for inp in inputs:
                    f.write(
                        json.dumps(
                            {
                                "key": inp.key,
                                "prompt": inp.prompt,
                                "response": responses[inp.key],
                            }
                        )
                        + "\n"
                    )

        strict, loose = score(inputs, responses)
        for mode, outputs in (("strict", strict), ("loose", loose)):
            s = summarize(outputs)
            repeat_summaries[mode].append(s)
            print(
                f"  {mode:6s} prompt-level={s['prompt_level']:.4f} " f"instruction-level={s['instruction_level']:.4f}"
            )
            evaluation_lib.write_outputs(
                str(out_dir / f"{safe_model}-eval_results_{mode}-r{r}.jsonl"),
                outputs,
            )

    # Aggregate across repeats (mean, like nemotron consensus reporting).
    final = {
        "model": args.model,
        "api_base": args.api_base,
        "input_file": args.input_file,
        "num_prompts": len(inputs),
        "num_repeats": args.num_repeats,
        "temperature": args.temperature,
        "results": {},
    }
    print("\n" + "=" * 64)
    print("FINAL (mean over repeats)")
    print("=" * 64)
    for mode in ("strict", "loose"):
        pl = [s["prompt_level"] for s in repeat_summaries[mode]]
        il = [s["instruction_level"] for s in repeat_summaries[mode]]
        entry = {
            "prompt_level_mean": statistics.mean(pl),
            "instruction_level_mean": statistics.mean(il),
            "prompt_level_per_repeat": pl,
            "instruction_level_per_repeat": il,
        }
        if len(pl) > 1:
            entry["prompt_level_std"] = statistics.stdev(pl)
            entry["instruction_level_std"] = statistics.stdev(il)
        final["results"][mode] = entry
        std = f" ± {entry['prompt_level_std']:.4f}" if len(pl) > 1 else ""
        print(
            f"{mode:6s} prompt-level={entry['prompt_level_mean']:.4f}{std}  "
            f"instruction-level={entry['instruction_level_mean']:.4f}"
        )

    results_path = out_dir / f"{safe_model}-results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved summary: {results_path}")


if __name__ == "__main__":
    main()
