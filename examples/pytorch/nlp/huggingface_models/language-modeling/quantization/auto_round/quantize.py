import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import logging
from auto_round import AutoRound

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


topologies_config = {
    "ds_mxfp8": {
        "scheme": "MXFP8",
        "fp_layers": "lm_head",
        "iters": 0,
    },
    "ds_mxfp4": {
        "scheme": "MXFP4",
        "fp_layers": "lm_head,self_attn",
        "iters": 0,
    },
    "qwen_mxfp8": {
        "scheme": "MXFP8",
        "fp_layers": "lm_head,mlp.gate",
        "iters": 0,
    },
    "qwen_mxfp4": {
        "scheme": "MXFP4",
        "fp_layers": "lm_head,mlp.gate,self_attn",
        "iters": 0,  # TODO: set to 200 before merge
    },
}


def quant_model(args):
    config = topologies_config[args.t]

    logger.info(f"Using fp_layers: {config['fp_layers']}")
    autoround = AutoRound(
        model=args.model,
        scheme=config["scheme"],
        enable_torch_compile=args.enable_torch_compile,
        iters=config["iters"],
        fp_layers=config["fp_layers"],
    )
    logger.info(f"Save quantized model to {args.output_dir}")
    format_type = "auto_round" if args.use_autoround_format else "llm_compressor"
    autoround.quantize_and_save(
        format=format_type,
        output_dir=f"{args.output_dir}/quantized_model_{args.t}",
    )


def get_model_and_tokenizer(model_name):
    # Load model and tokenizer
    fp32_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return fp32_model, tokenizer


def quant_model_(args):
    from neural_compressor.torch.quantization import (
        AutoRoundConfig,
        convert,
        prepare,
    )

    config = topologies_config[args.t]
    export_format = "auto_round" if args.use_autoround_format else "llm_compressor"
    output_dir = f"{args.output_dir}/quantized_model_{args.t}"
    fp32_model, tokenizer = get_model_and_tokenizer(args.model)
    quant_config = AutoRoundConfig(
        tokenizer=tokenizer,
        # nsamples=32,
        # seqlen=10,
        # iters=1,
        # amp=False,
        # scale_dtype="fp16",
        scheme=config["scheme"],
        enable_torch_compile=args.enable_torch_compile,
        iters=config["iters"],
        fp_layers=config["fp_layers"],
        export_format=export_format,
        output_dir=output_dir,
    )

    # quantizer execute
    model = prepare(model=fp32_model, quant_config=quant_config)
    inc_model = convert(model)
    logger.info(f"Quantized model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select a quantization scheme.")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the pre-trained model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "-t",
        type=str,
        choices=topologies_config.keys(),
        default="qwen_mxfp4",
        help="Quantization scheme to use. Available options: " + ", ".join(topologies_config.keys()),
    )

    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Enable torch compile for the model.",
    )
    parser.add_argument(
        "--use_autoround_format",
        action="store_true",
        help="Use AutoRound format for saving the quantized model.",
    )

    parser.add_argument(
        "--skip_attn",
        action="store_true",
        help="Skip quantize attention layers.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=0,
        help="Number of iterations for quantization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the quantized model.",
    )

    args = parser.parse_args()

    quant_model(args)
