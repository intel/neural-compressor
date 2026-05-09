# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copied from https://github.com/vllm-project/vllm/

from pathlib import Path

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


DEFAULT_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def resolve_prompts(args: dict) -> list[str]:
    prompt = args.pop("prompt", None)
    prompt_file = args.pop("prompt_file", None)

    if prompt is not None:
        return [prompt]

    if prompt_file is not None:
        prompt_path = Path(prompt_file).expanduser()
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Prompt file does not exist or is not a file: {prompt_path}"
            )
        return [prompt_path.read_text(encoding="utf-8")]

    return list(DEFAULT_PROMPTS)


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Single prompt text to generate from.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="Local file containing one prompt, useful for very long inputs.",
    )

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    prompts = resolve_prompts(args)

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
