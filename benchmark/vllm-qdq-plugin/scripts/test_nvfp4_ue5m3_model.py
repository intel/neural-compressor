"""Load an AutoRound NVFP4_E5M3 model and run one prompt with vLLM."""

import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--prompt",
        default="Briefly explain what quantization does in one sentence.",
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    output = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=32),
    )[0]
    print(f"PROMPT: {output.prompt}")
    print(f"OUTPUT: {output.outputs[0].text}")


if __name__ == "__main__":
    main()
