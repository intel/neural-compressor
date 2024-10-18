
if __name__ == "__main__":

    import sys

    sys.path.insert(0, '../../../')
    import time
    import torch
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="/models/opt-125m/"
    )
    parser.add_argument(
        "--eval_bs", default=4, type=int,
    )
    parser.add_argument(
        "--trust_remote_code", action='store_true',
        help="Whether to enable trust_remote_code"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device (e.g. cpu/cuda:0/hpu) for evaluation."
    )
    parser.add_argument(
        "--base_model", default="Qwen/Qwen-VL"
    )
    parser.add_argument(
        "--model_dtype", default=None, type=str,
        help="force to convert the dtype, some backends supports fp16 dtype better"
    )
    parser.add_argument(
        "--tasks",
        default="textvqa_val,scienceqa_test_img",
        help="lm-eval tasks for lm_eval version 0.4.2"
    )

    args = parser.parse_args()
    s = time.time()
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from auto_round.utils import convert_dtype_torch2str

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)

    if hasattr(config, "quantization_config"):
        quantization_config = config.quantization_config
        if "quant_method" in quantization_config and "auto-round" in quantization_config["quant_method"]:
            from auto_round.auto_quantizer import AutoHfQuantizer
        elif "quant_method" in quantization_config and quantization_config["quant_method"] == "gptq":
            if args.device == "hpu":
                from auto_round.auto_quantizer import AutoHfQuantizer
    model_name = args.model_name
    torch_dtype = torch.float
    if args.model_dtype != None:
        if args.model_dtype == "float16" or args.model_dtype == "fp16":
            torch_dtype = torch.float16
        if args.model_dtype == "bfloat16" or args.model_dtype == "bfp16":
            torch_dtype = torch.bfloat16
    dtype_str = convert_dtype_torch2str(torch_dtype)
    if dtype_str == "bf16":
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=args.trust_remote_code, device_map=args.device, bf16=True).eval()
    elif dtype_str == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=args.trust_remote_code, device_map=args.device, fp16=True).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=args.trust_remote_code, device_map=args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code, padding_side="right", use_fast=False)
    tokenizer.pad_token_id = tokenizer.eod_id
    test_tasks = args.tasks
    if isinstance(test_tasks, str):
        test_tasks = test_tasks.split(',')
    device = args.device
    for dataset in test_tasks:
        if 'vqa' in dataset:
            from evaluate_vqa import textVQA_evaluation
            with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch_dtype):
                evaluator = textVQA_evaluation(
                    model,
                    dataset_name=dataset,
                    # dataset_path=args.eval_path,
                    tokenizer=tokenizer,
                    batch_size=args.eval_bs,
                    trust_remote_code=args.trust_remote_code,
                    device=str(device)
                )
        elif 'scienceqa' in dataset:
            from evaluate_multiple_choice import scienceQA_evaluation
            with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch_dtype):
                evaluator = scienceQA_evaluation(
                    model,
                    dataset_name=dataset,
                    # dataset_path=args.eval_path,
                    tokenizer=tokenizer,
                    batch_size=args.eval_bs,
                    trust_remote_code=args.trust_remote_code,
                    device=str(device)
                )

    print("cost time: ", time.time() - s)



