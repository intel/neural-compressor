import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    # step 1
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--annotation-file", type=str, default="annotation.jsonl")
    parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ', 'GPTQ'],
                        help="Weight-only parameter.")
    parser.add_argument("--woq_bits", type=int, default=8)
    parser.add_argument("--woq_group_size", type=int, default=-1)
    parser.add_argument("--woq_scheme", default="sym")
    parser.add_argument("--woq_enable_mse_search", action="store_true")
    parser.add_argument("--woq_enable_full_range", action="store_true")
    # =============GPTQ configs====================
    parser.add_argument("--gptq_actorder", action="store_true",
                        help="Whether to apply the activation order GPTQ heuristic.")
    parser.add_argument('--gptq_percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--gptq_block_size', type=int, default=128, help='Block size. sub weight matrix size to run GPTQ.')
    parser.add_argument('--gptq_nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--gptq_use_max_length', action="store_true",
                        help='Set all sequence length to be same length of args.gptq_pad_max_length')
    parser.add_argument('--gptq_pad_max_length', type=int, default=2048, help='Calibration dataset sequence max length, \
                                                                            this should align with your model config, \
                                                                            and your dataset builder args: args.pad_max_length')
    parser.add_argument('--gptq_static_groups', action='store_true', help='Use determined group to do quantization')
    # ==============code generation args===========
    args = parser.parse_args()

    if args.quantize:
        # dataset
        user_model, tokenizer = get_user_model()
        calib_dataset = load_dataset(args.dataset, split="train")
        # calib_dataset = datasets.load_from_disk('/your/local/dataset/pile-10k/') # use this if trouble with connecting to HF
        calib_dataset = calib_dataset.shuffle(seed=args.seed)
        calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=args.pad_max_length, is_calib=True)
        calib_dataloader = DataLoader(
            calib_evaluator.dataset,
            batch_size=calib_size,
            shuffle=False,
            collate_fn=calib_evaluator.collate_batch,
        )


        def calib_func(prepared_model):
            for i, calib_input in enumerate(calib_dataloader):
                if i > args.calib_iters:
                    break
                prepared_model(calib_input[0])


        recipes = {}
        eval_func = None
        from neural_compressor import PostTrainingQuantConfig, quantization

        # specify the op_type_dict and op_name_dict
        if args.approach == 'weight_only':
            op_type_dict = {
                '.*': {  # re.match
                    "weight": {
                        'bits': args.woq_bits,  # 1-8 bits
                        'group_size': args.woq_group_size,  # -1 (per-channel)
                        'scheme': args.woq_scheme,  # sym/asym
                        'algorithm': args.woq_algo,  # RTN/AWQ/TEQ
                    },
                },
            }
            op_name_dict = {
                'lm_head': {"weight": {'dtype': 'fp32'}, },
                'embed_out': {"weight": {'dtype': 'fp32'}, },  # for dolly_v2
            }
            recipes["rtn_args"] = {
                "enable_mse_search": args.woq_enable_mse_search,
                "enable_full_range": args.woq_enable_full_range,
            }
            # gptq_true_sequential_test = [
            #     ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            #     ['self_attn.o_proj'],
            #     ['mlp.up_proj', 'mlp.gate_proj'],
            #     ['mlp.down_proj']
            # ]
            recipes['gptq_args'] = {
                'percdamp': args.gptq_percdamp,
                'act_order': args.gptq_actorder,
                'block_size': args.gptq_block_size,
                'nsamples': args.gptq_nsamples,
                'use_max_length': args.gptq_use_max_length,
                'pad_max_length': args.gptq_pad_max_length,
                'static_groups': args.gptq_static_groups,
                "true_sequential": args.gptq_true_sequential,
                "lm_head": args.gptq_lm_head,
            }
            # GPTQ: use assistive functions to modify calib_dataloader and calib_func
            # TEQ: set calib_func=None, use default training func as calib_func
            if args.woq_algo in ["GPTQ", "TEQ"]:
                calib_func = None

            conf = PostTrainingQuantConfig(
                approach=args.approach,
                op_type_dict=op_type_dict,
                op_name_dict=op_name_dict,
                recipes=recipes,
            )
        else:
            if re.search("gpt", user_model.config.model_type):
                op_type_dict = {
                    "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
                }
            else:
                op_type_dict = {}
            excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
            if args.sq:
                # alpha can be a float number of a list of float number.
                args.alpha = args.alpha if args.alpha == "auto" else eval(args.alpha)
                if re.search("falcon", user_model.config.model_type):
                    recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha, 'folding': False}}
                else:
                    recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha}}

            conf = PostTrainingQuantConfig(
                backend="ipex" if args.ipex else "default",
                approach=args.approach,
                excluded_precisions=excluded_precisions,
                op_type_dict=op_type_dict,
                recipes=recipes,
            )

            # eval_func should be set when tuning alpha.
            if isinstance(args.alpha, list):
                eval_dataset = load_dataset('lambada', split='validation')
                evaluator = Evaluator(eval_dataset, tokenizer)

                def eval_func(model):
                    acc = evaluator.evaluate(model)
                    return acc

        DEV = torch.device("cuda:0")
        user_model.to(DEV)

        q_model = quantization.fit(
            user_model,
            conf,
            calib_dataloader=calib_dataloader,
            calib_func=calib_func,
            eval_func=eval_func,
        )

        q_model.save(args.output_dir)

    # === eval textvqa ===
    # import pdb;pdb.set_trace()
    from evaluator import TextVQAEvaluator
    
    # load the data to run gptq

    # evaluation
    eval_runner = TextVQAEvaluator(question_file = args.question_file, image_folder = args.image_folder)
    eval_runner.run_inference(args.model_name_or_path, args.answer_file)
    eval_runner.calcualate_benchmark(args.answer_file, args.annotation_file)

    # === eval pope ===
    from evaluator import POPEEvaluator
    # eval_runner = POPEEvaluator(question_file = args.question_file, image_folder = args.image_folder)
    # eval_runner.run_inference(args.model_name_or_path, args.answer_file)
    # eval_runner.calcualate_benchmark(args.question_file, args.answer_file, args.annotation_file)
    # eval_single_textvqa(args.annotation_file, args.result_file)
    # eval_single_pope(args.question_file, args.annotation_dir, args.result_file)

if __name__ == "__main__":
    main()
