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
from neural_compressor.torch.utils import is_hpex_available, world_size
from auto_round import AutoRound

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
    # shard model for multi-cards and enable hpu graph

    if world_size > 1:
        ds_inference_kwargs = {
            "dtype": config.torch_dtype,
            "tensor_parallel": {"tp_size": world_size},
        }
        import deepspeed

        ds_model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = ds_model.module
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 quantization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name or path"
    )
    parser.add_argument("--dtype", type=str, default="MXFP4", choices=["MXFP4", "MXFP8", "NVFP4", "NVFP4+", "uNVFP4"], help="data type")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize model")
    parser.add_argument("--device_map", type=str, default=None, help="device map for model")
    parser.add_argument("--use_recipe", action="store_true", help="whether to use recipe to quantize model")
    parser.add_argument("--recipe_file", type=str, default="recipes/Meta-Llama-3.1-8B-Instruct_6bits.json", help="path of recipe file")
    parser.add_argument("--mem_per_param_scale", default=13, type=int, help="memory per param scale factor")
    parser.add_argument("--iters", default=200, type=int, help="iters for autoround.")
    parser.add_argument("--seqlen", default=2048, type=int, help="sequence length for autoround.")
    parser.add_argument("--nsamples", default=128, type=int, help="number of samples for autoround.")
    parser.add_argument("--save", action="store_true", help="whether to save the quantized model")
    parser.add_argument("--save_path", type=str, default="saved_results", help="path to save the quantized model")
    parser.add_argument("--save_format", type=str, default="auto_round", help="format to save the quantized model")
    parser.add_argument("--enable_torch_compile", action="store_true", help="whether to enable torch.compile")
    parser.add_argument("--quant_lm_head", action="store_true", help="whether to quantize lm_head")
    parser.add_argument("--accuracy", action="store_true", help="accuracy measurement")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for accuracy evaluation.")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "piqa",
            "hellaswag",
            "mmlu",
            "winogrande",
            "lambada_openai",
        ],
        help="tasks for accuracy validation, text-generation and code-generation tasks are different.",
    )
    parser.add_argument("--limit", type=int, default=None, help="number of samples for accuracy evaluation")
    args = parser.parse_args()

    print("Target data type:", args.dtype)

    model, tokenizer = initialize_model_and_tokenizer(args.model_name_or_path)
    device="hpu" if is_hpex_available() else "cuda"

    if args.quantize:
        if args.dtype in ["uNVFP4", "NVFP4+"]:
            from auto_round.schemes import QuantizationScheme

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

        if args.quant_lm_head:
            layer_config = {"lm_head": args.dtype}

        autoround = AutoRound(
            model,
            tokenizer,
            device=device,
            device_map="tp" if world_size > 1 else args.device_map,
            iters=args.iters,
            seqlen=args.seqlen,
            nsamples=args.nsamples,
            low_gpu_mem_usage=True,
            scheme=args.dtype,
            layer_config=layer_config if args.quant_lm_head else None,
            enable_torch_compile=args.enable_torch_compile,
            mem_per_param_scale=args.mem_per_param_scale,
        )

        if args.use_recipe:
            ############ load recipe results (MXFP4 + MXFP8) ############
            def load_recipe_results(file_path):
                import json
                with open(file_path, "r") as f:
                    return json.load(f)

            layer_config = load_recipe_results(args.recipe_file)
            if args.quant_lm_head:
                # ensure lm_head is quantized with mxfp8_config
                layer_config.update({"lm_head": "MXFP8"})
                print("In recipe mode, lm_head is quantized with MXFP8.")
            autoround.layer_config = layer_config

        # A placeholder, to pass assertion in AutoRound
        autoround.formats = "auto_round"
        autoround.quantize()
        model = autoround.model

    if args.accuracy:
        # set dtype to BF16 for HPU inference performance
        model = model.to(torch.bfloat16)
        model = model.eval().to(device)
        if is_hpex_available():
            # HPU needs padding to buckets for better performance
            # Generation tasks, such as gsm8k and mmlu-pro, may get OOM.
            model = wrap_in_hpu_graph(model)
            htcore.hpu_inference_initialize(model, mark_only_scales_as_const=True)
            from neural_compressor.evaluation.lm_eval import LMEvalParser, evaluate

            tasks = ",".join(args.tasks)
            eval_args = LMEvalParser(
                model="hf",
                user_model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                tasks=tasks,
                device="hpu",
                pad_to_buckets=True,
                limit=args.limit,
                add_bos_token=True,
            )
            results = evaluate(eval_args)
            torch.hpu.synchronize()
            all_accuracy = {}
            for task_name, task_results in results["results"].items():
                if task_name in ["hellaswag", "lambada_openai", "piqa", "winogrande", "mmlu"]:
                    accu = task_results["acc,none"]
                    all_accuracy[task_name] = accu
                    print(f"Accuracy for {task_name}: {accu:.4f}")
            print(f"Overall accuracy: {sum(all_accuracy.values())/len(all_accuracy):.4f}")
        else:
            # CUDA evaluation support all tasks.
            # gsm8k requires add_bos_token=False for better accuracy for llama model.
            args.tasks = ["piqa", "hellaswag", "mmlu", "gsm8k"]
            all_accuracy = {}
            test_gsm8k = False
            test_normal = False
            if "gsm8k" in args.tasks:
                test_gsm8k = True
                args.tasks.remove("gsm8k")
            if args.tasks:
                test_normal = True
            import lm_eval
            from lm_eval.models.huggingface import HFLM

            ########################## gms8k (ahead of normal tasks) #########################
            if test_gsm8k:
                lm = HFLM(
                    pretrained=model,
                    tokenizer=tokenizer,
                    add_bos_token=False,
                    batch_size=args.batch_size,
                )
                results_gsm8k = lm_eval.simple_evaluate(
                    lm,
                    tasks=["gsm8k"],
                    limit=args.limit,
                )
                for task_name, task_results in results_gsm8k["results"].items():
                    accu = task_results["exact_match,strict-match"]
                    all_accuracy[task_name] = accu
            ########################## gms8k end #########################
            if test_normal:
                lm = HFLM(
                    pretrained=model,
                    tokenizer=tokenizer,
                    add_bos_token=True,
                    batch_size=args.batch_size,
                )
                results = lm_eval.simple_evaluate(
                    lm,
                    tasks=args.tasks,
                    limit=args.limit,
                )
                for task_name, task_results in results["results"].items():
                    if task_name in ["hellaswag", "lambada_openai", "piqa", "winogrande", "mmlu"]:
                        accu = task_results["acc,none"]
                        all_accuracy[task_name] = accu
            for task_name, accu in all_accuracy.items():
                print(f"Accuracy for {task_name}: {accu:.4f}")
            print(f"Overall accuracy: {sum(all_accuracy.values())/len(all_accuracy):.4f}")

    if args.save:
        if args.dtype == "NVFP4":
            # using llm_compressor format to save nv_fp4 model
            autoround.save_quantized(args.save_path, format=args.save_format)
        else:
            # using auto_round format to save mx_fp4 and mx_fp8 model
            if world_size > 1:
                print(f"Suggest to save model without sharding for better reload experience.")
                print(f"Setting`--device_map 0,1,2,3` provides pipeline parallel instead of deepspeed tensor parallel.")
                output_dir = args.save_path + "/" + args.local_rank + "_" + args.world_size
                autoround.save_quantized(output_dir, format=args.save_format)
            else:
                autoround.save_quantized(args.save_path, format=args.save_format)
        print(f"Quantized model in {args.save_format} format is saved to {args.save_path}")
