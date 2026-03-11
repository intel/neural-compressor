# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import copy

import torch
import transformers

# For reproducibility
torch.manual_seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)
######################## HPU Memory Optimization ###########################
# ensure that unnecessary memory is released during quantization.
os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")
if int(os.getenv("WORLD_SIZE", "0")) > 0:
    os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
    os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
from neural_compressor.torch.utils import is_hpex_available
from neural_compressor.torch.quantization import autotune, prepare, convert, AutoRoundConfig, TuningConfig

if is_hpex_available():
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph

    htcore.hpu_set_env()
############################################################################


def initialize_model_and_tokenizer(model_name_or_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    # using memory mapping with torch_dtype=config.torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=config.torch_dtype)
    model.eval()
    return model, tokenizer


def dispatch_model_on_devices(model):
    from accelerate.big_modeling import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_max_memory, get_balanced_memory

    no_split_modules = getattr(model, "_no_split_modules", [])
    balanced_memory = get_balanced_memory(model)  # to initialize the function cache
    auto_device_map = infer_auto_device_map(
        model,
        max_memory=balanced_memory,
        no_split_module_classes=no_split_modules
    )
    print(auto_device_map)
    model = dispatch_model(model, auto_device_map)
    return model



@torch.no_grad()
def get_accuracy(model_name_or_path, tokenizer=None, eval_tasks="mmlu", limit=None):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    all_accuracy = {}
    special_tasks = []
    normal_tasks = []
    # Identify special tasks
    for t in eval_tasks:
        if t in ["gsm8k_llama", "mmlu_llama"]:
            special_tasks.append(t)
        else:
            normal_tasks.append(t)
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model_name_or_path,
        tokenizer=tokenizer,
        add_bos_token=True,
        batch_size=args.eval_batch_size,
    )
    # Run special tasks with chat template
    for special_task in special_tasks:
        results_special = lm_eval.simple_evaluate(
            lm,
            tasks=[special_task],
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            limit=args.limit if limit is None else limit,
        )
        for task_name, task_results in results_special["results"].items():
            # gsm8k_llama uses exact_match,strict-match, mmlu_llama may use acc,none
            if task_name in special_tasks:
                if "exact_match,strict_match" in task_results:
                    accu = task_results["exact_match,strict_match"]
                elif "acc,none" in task_results:
                    accu = task_results["acc,none"]
                else:
                    accu = list(task_results.values())[0]
                all_accuracy[task_name] = accu

    # Run normal tasks without chat template
    if normal_tasks:
        results = lm_eval.simple_evaluate(
            lm,
            tasks=normal_tasks,
            limit=args.limit if limit is None else limit,
        )
        for task_name, task_results in results["results"].items():
            if "acc,none" in task_results and task_name in normal_tasks:
                accu = task_results["acc,none"]
                all_accuracy[task_name] = accu
    for task_name, accu in all_accuracy.items():
        print(f"Accuracy for {task_name}: {accu:.4f}")
    avg_accu = sum(all_accuracy.values())/len(all_accuracy)
    print(f"Overall accuracy: {avg_accu:.4f}")
    return avg_accu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 quantization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name or path"
    )
    parser.add_argument("--dtype", type=str, default="MXFP4", choices=["MXFP4", "MXFP8", "NVFP4", "NVFP4+", "uNVFP4"], help="data type")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize model")
    parser.add_argument("--device_map", type=str, default="auto", help="device map for model")
    parser.add_argument(
        "--target_bits",
        type=float,
        nargs="+",
        default=None, 
        help="target bits for mix precision"
    )
    parser.add_argument("--tolerable_loss", type=float, default=0.01, 
            help="tolerable loss for accuracy autotune, relative value to the fp32 baseline")
    parser.add_argument(
        "--options",
        type=str,
        nargs="+",
        default=[
            "MXFP4",
            "MXFP8",
        ],
        help="options for mix precision"
    )
    parser.add_argument(
        "--shared_layers",
        type=str,
        nargs="+",
        action='append',
        default=[],
        help="[mix-precision] ensure that listed layers are using same data type for quantization"
    )
    parser.add_argument(
        "--static_kv_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn"],
        help="Data type for static quantize key and value.",
    )
    parser.add_argument(
        "--static_attention_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn"],
        help="Data type for static quantize key and value.",
    )
    parser.add_argument("--use_recipe", action="store_true", help="whether to use recipe to quantize model")
    parser.add_argument("--recipe_file", type=str, default="recipes/Meta-Llama-3.1-8B-Instruct_6bits.json", help="path of recipe file")
    parser.add_argument("--iters", default=200, type=int, help="iters for autoround.")
    parser.add_argument("--seqlen", default=2048, type=int, help="sequence length for autoround.")
    parser.add_argument("--nsamples", default=128, type=int, help="number of samples for autoround.")
    parser.add_argument("--save", action="store_true", help="whether to save the quantized model")
    parser.add_argument("--export_path", type=str, default="saved_results", help="path to save the quantized model")
    parser.add_argument("--export_format", type=str, default="auto_round", help="format to save the quantized model")
    parser.add_argument("--enable_torch_compile", action="store_true", help="whether to enable torch.compile")
    parser.add_argument("--low_gpu_mem_usage", action="store_true", help="whether to enable low_gpu_mem_usage")
    parser.add_argument("--quant_lm_head", action="store_true", help="whether to quantize lm_head")
    parser.add_argument("--accuracy", action="store_true", help="accuracy measurement")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="batch size for accuracy evaluation.")
    parser.add_argument(
        "--tune_tasks",
        type=str,
        nargs="+",
        default=None,
        help="tasks for accuracy validation of autotune, text-generation and code-generation tasks are different.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "piqa",
            "hellaswag",
            "mmlu_llama",
            "gsm8k_llama",
        ],
        help="tasks for accuracy validation, text-generation and code-generation tasks are different.",
    )
    parser.add_argument("--limit", type=int, default=None, help="number of samples for accuracy evaluation")
    parser.add_argument("--tune_limit", type=int, default=None, help="number of samples for accuracy autotune")
    args = parser.parse_args()

    if args.target_bits is None:
        print("Target data type:", args.dtype)
    else:
        print("Target data type for mix precision:", args.options)
        print("Layers sharing the same data type:", args.shared_layers)
    model, tokenizer = initialize_model_and_tokenizer(args.model_name_or_path)

    if args.quantize:
        from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme

        # Check if RCEIL versions are available and use them instead
        use_rceil = "MXFP4_RCEIL" in PRESET_SCHEMES and "MXFP8_RCEIL" in PRESET_SCHEMES
        if use_rceil:
            # Replace dtype if it's MXFP4 or MXFP8
            if args.dtype == "MXFP4":
                args.dtype = "MXFP4_RCEIL"
            elif args.dtype == "MXFP8":
                args.dtype = "MXFP8_RCEIL"
            # Replace options list entries
            args.options = [
                "MXFP4_RCEIL" if opt == "MXFP4" else ("MXFP8_RCEIL" if opt == "MXFP8" else opt)
                for opt in args.options
            ]

        if args.dtype in ["uNVFP4", "NVFP4+"]:
            uNVFP4 = QuantizationScheme.from_dict(
                {
                    "bits": 4,
                    "group_size": 16,
                    "data_type": "fp4_v2",
                    "act_bits": 4,
                    "act_data_type": "fp4_v2",
                    "act_group_size": 16,
                    "act_sym": True,
                }
            )
            args.dtype = uNVFP4

        layer_config = {}
        if args.use_recipe:
            ############ load recipe results (MXFP4 + MXFP8) ############
            def load_recipe_results(file_path):
                import json
                with open(file_path, "r") as f:
                    return json.load(f)

            layer_config = load_recipe_results(args.recipe_file)
        if args.quant_lm_head:
            # ensure lm_head is quantized with mxfp8_config
            layer_config.update({"lm_head": args.dtype})

        # preprocess
        if isinstance(args.target_bits, list) and len(args.target_bits) == 1:
            args.target_bits = args.target_bits[0]
        config = AutoRoundConfig(
            tokenizer=tokenizer,
            iters=args.iters,
            seqlen=args.seqlen,
            nsamples=args.nsamples,
            scheme=args.dtype,
            target_bits=args.target_bits,
            options=args.options,
            shared_layers=args.shared_layers,
            static_kv_dtype=args.static_kv_dtype,
            static_attention_dtype=args.static_attention_dtype,
            enable_torch_compile=args.enable_torch_compile,
            low_gpu_mem_usage=args.low_gpu_mem_usage,
            export_format=args.export_format,
            output_dir=args.export_path,
            device_map=args.device_map,
            layer_config=layer_config if (args.use_recipe or args.quant_lm_head) else None,
        )
        if isinstance(args.target_bits, list) and len(args.target_bits) > 1:
            args.tune_tasks = args.tasks if args.tune_tasks is None else args.tune_tasks

            def eval_fn(model):
                model = model.eval()
                model = dispatch_model_on_devices(model)
                accu = get_accuracy(model, tokenizer, args.tune_tasks, args.tune_limit)
                model = model.to("cpu")
                return accu
            tuning_config = TuningConfig(config_set=[config], tolerable_loss=args.tolerable_loss)
            model = autotune(model, tuning_config, eval_fn=eval_fn)
        else:
            model = prepare(model, config)
            model = convert(model)
        print(f"Quantized model in {args.export_format} format is saved to {args.export_path}")

    if args.accuracy:
        model = dispatch_model_on_devices(model)
        get_accuracy(model, tokenizer, args.tasks)
