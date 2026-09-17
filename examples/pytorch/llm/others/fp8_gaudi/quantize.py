import os
import argparse
import tqdm

# ensure that unnecessary memory is released during quantization.
os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")
if int(os.getenv("WORLD_SIZE", "0")) > 0:
    os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
    os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")


import torch
import habana_frameworks.torch.core as htcore

from neural_compressor.torch.quantization import (
    FP8Config,
    prepare,
    convert,
    finalize_calibration,
    save,
    load,
)
from neural_compressor.torch.utils import get_used_hpu_mem_MB, get_used_cpu_mem_MB, logger, forward_wrapper
from neural_compressor.torch.utils.block_wise import block_wise_calibration
from neural_compressor.torch.utils.llm_utility import (
    initialize_model_and_tokenizer,
    get_default_llm_dataloader,
    llm_benchmark,
)

# use no_grad mode for quantization
torch.set_grad_enabled(False)
htcore.hpu_set_env()
hpu_mem_0 = get_used_hpu_mem_MB()
cpu_mem_0 = get_used_cpu_mem_MB()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 quantization.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-405B", help="model name or path")
    parser.add_argument("--quantize", action="store_true", help="whether to quantize model")
    parser.add_argument("--scale_method", type=str, default="maxabs_hw", help="Choose scale method", choices=[
        # per-tensor
        "unit_scale", "hw_aligned_single_scale", "maxabs_hw", "maxabs_pow2", 
        "maxabs_arbitrary", "maxabs_hw_opt_weight", "maxabs_pow2_opt_weight", 
        # per-channel
        "act_maxabs_hw_weights_pcs_maxabs_pow2", "act_maxabs_hw_weights_pcs_opt_pow2", 
        "act_maxabs_pow2_weights_pcs_maxabs_pow2", "act_maxabs_pow2_weights_pcs_opt_pow2",
    ])
    parser.add_argument("--use_hpu_graph", action="store_true", help="whether to use hpu graph mode to accelerate performance")
    parser.add_argument("--enable_block_wise_calibration", action="store_true", help="whether to use block-wise calibration")
    parser.add_argument("--disable_optimum_habana", action="store_true", help="whether to use adapt_transformers_to_gaudi")
    parser.add_argument("--save", action="store_true", help="whether to save the quantized model")
    parser.add_argument("--load", action="store_true", help="whether to load the quantized model")
    parser.add_argument("--save_path", type=str, default="saved_results", help="path to save the quantized model")
    parser.add_argument("--accuracy", action="store_true", help="accuracy measurement")
    parser.add_argument("--performance", action="store_true", help="performance measurement")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for accuracy measurement.")
    parser.add_argument("--num_fewshot", default=0, type=int, help="num_fewshot of lm_eval.")
    parser.add_argument("--dump_stats_path", type=str, default="./hqt_output/measure", help="path and prefix to calibration info file.")
    parser.add_argument("--tasks", default="lambada_openai",
                        type=str, help="tasks for accuracy validation, text-generation and code-generation tasks are different.")
    parser.add_argument("--dataset_name", type=str, default="NeelNanda/pile-10k", help="dataset name for calibration dataloader")
    parser.add_argument("--nsamples", type=int, default=128, help="number of samples for calibration dataloader")
    parser.add_argument("--seq_len", type=int, default=128, help="sequence length for calibration dataloader and benchmarking")
    args = parser.parse_args()
    if not args.disable_optimum_habana:
        # Tweak generation so that it runs faster on Gaudi
        import transformers
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        if args.quantize:
            orig_check_support_param_buffer_assignment = transformers.modeling_utils.check_support_param_buffer_assignment
            adapt_transformers_to_gaudi()
            # to protect memory mapping usage for quantization
            transformers.modeling_utils.check_support_param_buffer_assignment = orig_check_support_param_buffer_assignment
        else:
            adapt_transformers_to_gaudi()

    model, tokenizer = initialize_model_and_tokenizer(args.model_name_or_path, use_load=args.load, device="hpu")
    # show used memory
    logger.info(f"After loading model, used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
    logger.info(f"After loading model, used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")

    if args.quantize:
        if args.enable_block_wise_calibration:
            logger.warning("Block-wise calibration is enabled, lm_head will be excluded from calibration.")

        # prepare
        qconfig = FP8Config(
            fp8_config="E4M3",
            scale_method=args.scale_method,
            blocklist={"names": ["lm_head"]} if args.enable_block_wise_calibration else {},  # block-wise cannot calibrate lm_head
            measure_on_hpu=False if args.enable_block_wise_calibration else True,  # to avoid device mapping of model
            dump_stats_path=args.dump_stats_path,
        )
        if args.scale_method in ["unit_scale", "hw_aligned_single_scale"]:
            model = convert(model, qconfig)
        else:
            model = prepare(model, qconfig)

            # calibration
            dataloader = get_default_llm_dataloader(
                tokenizer, 
                dataset_name=args.dataset_name, 
                bs=args.batch_size, 
                nsamples=args.nsamples,
                seq_len=args.seq_len, 
                seed=42, 
            )
            if args.enable_block_wise_calibration:
                block_wise_calibration(model, dataloader)
            else:
                if args.use_hpu_graph:
                    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
                    model = wrap_in_hpu_graph(model)
                for data in tqdm.tqdm(dataloader):
                    logger.info("Calibration started")
                    forward_wrapper(model, data)
                    logger.info("Calibration end")

            # convert
            model = convert(model)

        # show used memory
        logger.info(f"Used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")
        if args.save:
            logger.info(f"Saving quantized model to {args.save_path}")
            save(model, args.save_path, format="huggingface")
            tokenizer.save_pretrained(args.save_path)
            logger.info(f"Saved quantized model to {args.save_path}")
        exit(0)  # model is wrapped during calibration, need to exit before accuracy and performance measurement

    # preprocess model for accuracy and performance measurement
    if not args.load:
        # compare fp8 with bf16, not fp32.
        model = model.to(torch.bfloat16)
    model = model.eval().to("hpu")
    if args.use_hpu_graph:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        model = wrap_in_hpu_graph(model)
    htcore.hpu_inference_initialize(model, mark_only_scales_as_const=True)

    if args.accuracy:
        from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
        eval_args = LMEvalParser(
            model="hf", 
            user_model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            tasks=args.tasks,
            device="hpu",
            pad_to_buckets=True,
            num_fewshot=args.num_fewshot,
        )
        results = evaluate(eval_args)
        # show used memory
        logger.info(f"Used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")


    if args.performance:
        llm_benchmark(model, args.batch_size, args.seq_len)
        # show used memory
        logger.info(f"Used HPU memory: {round((get_used_hpu_mem_MB() - hpu_mem_0)/1024, 3)} GiB")
        logger.info(f"Used CPU memory: {round((get_used_cpu_mem_MB() - cpu_mem_0)/1024, 3)} GiB")
