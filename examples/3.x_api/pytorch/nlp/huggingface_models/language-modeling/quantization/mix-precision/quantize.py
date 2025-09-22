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
    parser.add_argument("--dtype", type=str, default="mx_fp4", choices=["mx_fp4", "mx_fp8", "nv_fp2", "fp4_v2"], help="data type")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize model")
    parser.add_argument("--use_recipe", action="store_true", help="whether to use recipe to quantize model")
    parser.add_argument("--recipe_file", type=str, default="recipes/Meta-Llama-3.1-8B-Instruct_6bits.json", help="path of recipe file")
    parser.add_argument("--iters", default=200, type=int, help="iters for autoround.")
    parser.add_argument("--seqlen", default=2048, type=int, help="sequence length for autoround.")
    parser.add_argument("--nsamples", default=128, type=int, help="number of samples for autoround.")
    parser.add_argument("--save", action="store_true", help="whether to save the quantized model")
    parser.add_argument("--save_path", type=str, default="saved_results", help="path to save the quantized model")
    parser.add_argument("--quant_lm_head", action="store_true", help="whether to quantize lm_head")
    parser.add_argument("--accuracy", action="store_true", help="accuracy measurement")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size for autoround tuning.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="batch size for accuracy evaluation.")
    parser.add_argument(
        "--mxfp8_mod_list",
        type=str,
        nargs="*",
        default=[],  # 默认值
        help="List of module names or patterns for MXFP8 quantization.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",  # 接受一个或多个字符串作为列表
        default=[
            "piqa",
            "hellaswag",
            "mmlu",
            "winogrande",
            "lambada_openai",
        ],  # 默认值
        help="tasks for accuracy validation, text-generation and code-generation tasks are different.",
    )
    parser.add_argument("--limit", type=int, default=None, help="number of samples for accuracy evaluation")
    args = parser.parse_args()

    print("Target data type:", args.dtype)

    model, tokenizer = initialize_model_and_tokenizer(args.model_name_or_path)
    device="hpu" if is_hpex_available() else "cuda"

    if args.quantize:
        if args.quant_lm_head:
            lm_head_config = {
                "group_size": 32 if "mx" in args.dtype else 16,
                "data_type": args.dtype,
                "act_data_type": "fp4_v2_with_global_scale" if "fp4_v2" in args.dtype else args.dtype,
            }
            layer_config = {"lm_head": lm_head_config}

        autoround = AutoRound(
            model,
            tokenizer,
            device=device,
            device_map="tp" if world_size > 1 else None,
            iters=args.iters,
            seqlen=args.seqlen,
            nsamples=args.nsamples,
            batch_size=args.batch_size,
            low_gpu_mem_usage=True,
            group_size=32 if "mx" in args.dtype else 16,
            data_type=args.dtype,
            act_data_type="fp4_v2_with_global_scale" if "fp4_v2" in args.dtype else args.dtype,
            layer_config=layer_config if args.quant_lm_head else None,
        )

        if args.use_recipe:
            ############ load recipe results (MXFP4 + MXFP8) ############
            def load_recipe_results(file_path):
                import json
                with open(file_path, "r") as f:
                    return json.load(f)
                
            layer_config = load_recipe_results(args.recipe_file)
            if args.quant_lm_head:
                mxfp8_config = {
                    "bits": 8,
                    "group_size": 32,
                    "data_type": "mx_fp8",
                    "act_data_type": "mx_fp8",
                }
                # ensure lm_head is quantized with mxfp8_config
                layer_config.update({"lm_head": mxfp8_config})
                print("In recipe mode, lm_head is quantized with MXFP8.")
            autoround.layer_config = layer_config

        autoround.quantize()
        model = autoround.model

    # set dtype to BF16 for HPU inference performance
    model = model.to(torch.bfloat16)
    model = model.eval().to(device)

    print(model)

    if args.accuracy:
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
            # model = torch.compile(model)
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
