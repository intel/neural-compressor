import sys
sys.path.append("./")
import argparse
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
from tqdm import tqdm
import shortuuid
import math
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.train.train import preprocess, preprocess_multimodal

class CustomDataset(Dataset):
    # much refer to https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py
    def __init__(self, list_data_dict, image_folder, tokenizer, image_processor, args):
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args

    def __getitem__(self, index):
        sources = self.list_data_dict[index]

        # image
        image_file = os.path.basename(sources["image"])
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        sources = preprocess_multimodal(
            copy.deepcopy([sources["conversations"]]), # a list
            self.args,
        )

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[index])
        )
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        data_dict['image'] = image
        return data_dict["input_ids"], data_dict["image"], data_dict["image"].size()

    def __len__(self):
        return len(self.list_data_dict)

def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes

def create_data_loader(dataset, batch_size=1):
    assert batch_size == 1, "batch_size must be 1"
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn)
    return data_loader


#=====================

def get_user_argument():
    parser = argparse.ArgumentParser()
    # ================= Model ===================== 
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # ========== Calibration Datasets ============= 
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--is-multimodal", type=bool, default=False)
    parser.add_argument("--mm-use-im-start-end", type=bool, default=False)
    parser.add_argument("--calib_iters", default=512, type=int,
                        help="calibration iters.")
    # ======= General Quantization Choice =========
    parser.add_argument("--approach", type=str, default='static',
                        help="Select from ['dynamic', 'static', 'weight-only']")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ', 'GPTQ'],
                         help="Weight-only parameter.")
    parser.add_argument("--woq_bits", type=int, default=8)
    parser.add_argument("--woq_group_size", type=int, default=-1)
    parser.add_argument("--woq_scheme", default="sym")
    parser.add_argument("--woq_enable_mse_search", action="store_true")
    parser.add_argument("--woq_enable_full_range", action="store_true")
    # ============= GPTQ configs ==================
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
    parser.add_argument('--gptq_true_sequential', action='store_true', help="Whether to run in true_sequential model.")
    parser.add_argument('--gptq_multimodal', action='store_true', help='quantize a multimodal model')
    parser.add_argument('--gptq_lm_head', action='store_true', help="Whether to use GPTQ to quantize the output layer of the LLMs.")
    # ================= Evaluation Related =====================
    parser.add_argument("--eval-question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--eval-image-folder", type=str)
    parser.add_argument('--eval-result-file', type=str)
    parser.add_argument('--eval-annotation-file', type=str)
    args = parser.parse_args()
    return args

def main():
    disable_torch_init()
    # Step 1: load user defined arguments
    args = get_user_argument()

    # Step 2 load the unquantized model and calibration datasets (aligned with visual instruction tuning)
    from llava.model.builder import load_pretrained_model
    model_path = os.path.expanduser(args.model_name_or_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # load the quantization files
    questions = json.load(open(args.question_file, "r"))
    dataset = CustomDataset(questions, args.image_folder, tokenizer, image_processor, args)
    dataloader = create_data_loader(dataset)
    # import pdb;pdb.set_trace()
    # model.float()
    # model.eval()

    if False:
    # if args.quantize:
        from neural_compressor import PostTrainingQuantConfig, quantization
        recipes = {}
        eval_func = None

        def calib_func(prepared_model):
            for i, calib_input in enumerate(calib_dataloader):
                if i > args.calib_iters:
                    break
                prepared_model(calib_input[0])

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
            "multimodal": args.gptq_multimodal,
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
    
        q_model = quantization.fit(
            model, 
            conf,
            calib_dataloader = dataloader,
            calib_func = calib_func,
            eval_func = eval_func
        )

    # evaluation
    from mm_evaluation import TextVQAEvaluator
    evaluator = TextVQAEvaluator(
        model,
        tokenizer,
        image_processor,
        args.eval_image_folder,
        args.eval_question_file,
        args.eval_annotation_file,
        model_name = model_name
    )
    evaluator.run_evaluate(result_file = args.eval_result_file)
    evaluator.calculate_accuracy(result_file = args.eval_result_file)

if __name__ == "__main__":
    main()
