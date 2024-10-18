import argparse
parser = argparse.ArgumentParser()
import torch
import os
import transformers
# # os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True, warn_only=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import set_seed

import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
from llava.mm_utils import get_model_name_from_path
from llava.train.train import preprocess, preprocess_multimodal, DataCollatorForSupervisedDataset
from llava.model.builder import load_pretrained_model
from neural_compressor.torch.utils.utility import (get_multimodal_block_names,
                                                    get_layer_names_in_block,
                                                    detect_device,
                                                    run_fn_for_vlm_autoround
                                                    )
from neural_compressor.torch.quantization import (AutoRoundConfig,
                                                    prepare,
                                                    convert,
                                                    load)


def save_tower(model, save_path, quant_vision: bool = False, max_shard_size: str = "5GB", safe_serialization: bool = True):
    if not quant_vision:
        print("Won't save vision_tower since this part was not quantized.")
        return
    ori_path = save_path
    ori_tower_name = model.get_vision_tower().vision_tower_name
    vision_tower = model.get_vision_tower().vision_tower
    save_path = f'{save_path}-vision_tower'
    os.makedirs(save_path, exist_ok=True)
    quantization_config = model.config.quantization_config
    redundant_prefix = "model.vision_tower.vision_tower."
    org_block_list = copy.deepcopy(quantization_config['quant_block_list'])
    # prepare vision_tower quantize list
    quant_block_list = [element.split(redundant_prefix)[1] if redundant_prefix in element else "" \
                        for sublist in org_block_list for element in sublist]
    quant_block_list = [[element for element in quant_block_list if element != ""]]
    quantization_config['quant_block_list'] = quant_block_list
    if hasattr(vision_tower, "config"):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(ori_tower_name)
        processor.save_pretrained(save_path)
        vision_tower.config.quantization_config = quantization_config
        vision_tower.config.save_pretrained(save_path)
    vision_tower.save_pretrained(save_path, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
    # prepare llava model quantize list
    quant_block_list = [element if redundant_prefix not in element else "" \
                        for sublist in org_block_list for element in sublist]
    quant_block_list = [[element for element in quant_block_list if element != ""]]
    quantization_config['quant_block_list'] = quant_block_list
    model.config.mm_vision_tower = save_path
    model.config.save_pretrained(ori_path)
    

class CustomDataset(Dataset): # for llava tuning
    # much refer to https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py
    def __init__(self, list_data_dict, image_folder, tokenizer, image_processor, args):
        self.list_data_dict = list_data_dict
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.args.is_multimodal = args.is_multimodal

    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        # image = None
        image_file = os.path.basename(sources["image"])
        try:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        except Exception as error:
            print(f"{error}, skipped by set image to None")
            image = None
        sources = preprocess_multimodal(
            copy.deepcopy([sources["conversations"]]), # a list
            self.args,
        )
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[index]),
        )
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
        # image exist in the data
        data_dict['image'] = image
        return data_dict

    def __len__(self):
        return len(self.list_data_dict)


def create_data_loader(dataset, batch_size=1, data_collator=None):
    assert batch_size == 1, "batch_size must be 1"
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    return data_loader

if __name__ == '__main__':

    parser.add_argument("--model_name", default="liuhaotian/llava-v1.5-7b")
    
    parser.add_argument("--quantize", action="store_true")
    
    parser.add_argument("--accuracy", action="store_true")

    parser.add_argument("--bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--train_bs", default=1, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=4, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default="auto", type=str,
                        help="The device to be used for tuning. The default is set to auto/None,"
                             "allowing for automatic detection. Currently, device settings support CPU, GPU, and HPU.")

    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

    parser.add_argument("--iters", default=200, type=int,
                        help=" iters")

    parser.add_argument("--lr", default=None, type=float,
                        help="learning rate, if None, it will be set to 1.0/iters automatically")

    parser.add_argument("--minmax_lr", default=None, type=float,
                        help="minmax learning rate, if None,it will beset to be the same with lr")

    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    parser.add_argument("--eval_fp16_baseline", action='store_true',
                        help="whether to eval FP16 baseline")

    parser.add_argument("--adam", action='store_true',
                        help="adam")

    parser.add_argument("--seqlen", default=512, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

    parser.add_argument("--nblocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--nsamples", default=512, type=int,
                        help="number of samples")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="low_gpu_mem_usage is deprecated")

    parser.add_argument("--export_format", default='auto_round:gptq', type=str,
                        help="targeted inference acceleration platform,The options are 'fake', 'cpu', 'gpu', 'xpu' and 'auto_round'."
                             "default to 'fake', indicating that it only performs fake quantization and won't be exported to any device.")

    parser.add_argument("--scale_dtype", default='fp16',
                        help="which scale data type to use for quantization, 'fp16', 'fp32' or 'bf16'.")

    parser.add_argument("--output_dir", default="./tmp_autoround", type=str,
                        help="Where to store the final model.")

    parser.add_argument("--disable_eval", action='store_true',
                        help="Whether to do lmeval evaluation.")

    parser.add_argument("--disable_amp", action='store_true',
                        help="disable amp")

    parser.add_argument("--disable_minmax_tuning", action='store_true',
                        help="whether disable  enable weight minmax tuning")

    parser.add_argument("--disable_trust_remote_code", action='store_true',
                        help="Whether to disable trust_remote_code")

    parser.add_argument("--disable_quanted_input", action='store_true',
                        help="whether to disuse the output of quantized block to tune the next block")

    parser.add_argument("--quant_lm_head", action='store_true',
                        help="quant_lm_head")

    parser.add_argument("--model_dtype", default=None, type=str,
                        help="force to convert the dtype, some backends supports fp16 dtype better")
    
    parser.add_argument("--act_bits", default=32, type=int,
                    help="activation bits")
    
    parser.add_argument("--is_multimodal", type=bool, default=True,
                        help="To determine whether the preprocessing should handle multimodal infomations.")
    
    parser.add_argument("--quant_vision", action='store_true',
                        help="To determine whether the quantization should handle vision component.")
    
    # ========== Calibration Datasets ============= 
    parser.add_argument("--mm-use-im-start-end", type=bool, default=False)
    
    parser.add_argument("--image_folder", default="coco", type=str,
                        help="The dataset for quantization training. It can be a custom one.")
    
    parser.add_argument("--question_file", default=None, type=str,
                            help="The dataset for quantization training. It can be a custom one.")
    
    # ================= Evaluation Related =====================
    parser.add_argument("--eval_question_file", type=str, default="tables/question.jsonl")
    
    parser.add_argument("--eval_image_folder", type=str)
    
    parser.add_argument('--eval_result_file', type=str, default="./tmp_results")
    
    parser.add_argument('--eval_annotation_file', type=str)

    args = parser.parse_args()

    if args.quantize:
        set_seed(args.seed)

        if args.act_bits <= 8:
            print(
                "Warning, activation quantization is an experiment feature")
        
        if args.act_bits <= 8 and args.export_format != "fake":
            assert False, "only support fake mode for activation quantization currently"
            
        if "marlin" in args.export_format and args.sym == False:
            assert False, "marlin backend only supports sym quantization, please set --sym"
            
        model_name = args.model_name
        if model_name[-1] == "/":
            model_name = model_name[:-1]
        print(model_name, flush=True)

        device_str = detect_device(args.device)
        torch_dtype = "auto"
        torch_device = torch.device(device_str)
        model_path = args.model_name
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base=None, model_name=model_name,
                torch_dtype=torch_dtype)

        model = model.eval()

        if args.model_dtype != None:
            if args.model_dtype == "float16" or args.model_dtype == "fp16":
                model = model.to(torch.float16)
            if args.model_dtype == "bfloat16" or args.model_dtype == "bfp16":
                model = model.to(torch.bfloat16)
                
        seqlen = args.seqlen
        if hasattr(tokenizer, "model_max_length"):
            if tokenizer.model_max_length < seqlen:
                print(f"change sequence length to {tokenizer.model_max_length} due to the limitation of model_max_length",
                    flush=True)
                seqlen = min(seqlen, tokenizer.model_max_length)
                args.seqlen = seqlen

        excel_name = f"{model_name}_{args.bits}_{args.group_size}"
        pt_dtype = torch.float16
        if (hasattr(model, 'config') and (model.dtype is torch.bfloat16 or model.config.torch_dtype is torch.bfloat16)):
            dtype = 'bfloat16'
            pt_dtype = torch.bfloat16
        else:
            if str(args.device) != "cpu":
                pt_dtype = torch.float16
                dtype = 'float16'
            else:
                pt_dtype = torch.float32
                dtype = 'float32'

        questions = json.load(open(args.question_file, "r"))
        dataset = CustomDataset(questions, args.image_folder, tokenizer, image_processor, args=args)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        dataloader = create_data_loader(dataset, args.train_bs, data_collator)

        quant_block_list = get_multimodal_block_names(model, args.quant_vision)
            
        quant_config = AutoRoundConfig(bits=args.bits, use_sym=args.sym, batch_size=args.train_bs, group_size=args.group_size,
                            seqlen=seqlen, nblocks=args.nblocks, iters=args.iters, lr=args.lr,
                            minmax_lr=args.minmax_lr, enable_quanted_input=not args.disable_quanted_input,
                            nsamples=args.nsamples, seed=args.seed, gradient_accumulate_steps=args.gradient_accumulate_steps,
                            scale_dtype=args.scale_dtype, enable_minmax_tuning=not args.disable_minmax_tuning, act_bits=args.act_bits,
                            quant_block_list=quant_block_list, export_format=args.export_format)
        
        all_block_list = get_multimodal_block_names(model, quant_vision=True)
        all_block_set = set(tuple(block) for block in all_block_list)
        quant_block_set = set(tuple(block) for block in quant_block_list)
        set_to_full_prec = list(all_block_set - quant_block_set)
        set_to_full_prec = get_layer_names_in_block(model, quant_block_list=set_to_full_prec)
        for name in set_to_full_prec:
            quant_config.set_local(name, AutoRoundConfig(dtype="fp32"))
            
        # skip special layers
        quant_config.set_local("model.mm_projector*", AutoRoundConfig(dtype="fp32"))
            
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
                if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                    quant_config.set_local(n, AutoRoundConfig(dtype="fp32"))
                    print(
                        f"{n} will not be quantized due to its shape not being divisible by 32, resulting in an exporting issue to autogptq")
        
        lm_head_layer_name = "lm_head"
        if args.quant_lm_head:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
            if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
                tied_keys = model._tied_weights_keys
                for item in tied_keys:
                    if lm_head_layer_name in item:  ##TODO extend to encoder-decoder layer, seq classification model
                        args.quant_lm_head = False
                        print(
                            f"warning, disable quant_lm_head as quantizing lm_head with tied weights has not been "
                            f"supported currently")
                        break
                    
        if not args.quant_lm_head:
                quant_config.set_local(lm_head_layer_name, AutoRoundConfig(dtype="fp32"))
                transformers_version = [int(item) for item in transformers.__version__.split('.')[:2]]
                if transformers_version[0] == 4 and transformers_version[1] < 38:
                    error_message = "Please upgrade transformers>=4.38.0 to support lm-head quantization."
                    raise EnvironmentError(error_message)
        
        run_args = (dataloader, seqlen, args.nsamples)
        user_model = prepare(model=model, quant_config=quant_config)
        run_fn_for_vlm_autoround(user_model, *run_args)
        user_model = convert(user_model)

        from neural_compressor.torch.utils import LoadFormat
        save_tower(user_model, args.output_dir, quant_vision=args.quant_vision)
        user_model.save(args.output_dir, format=LoadFormat.HUGGINGFACE)
        if tokenizer is not None:
            tokenizer.save_pretrained(args.output_dir)

    if args.accuracy:
        device_str = detect_device(args.device)
        torch_device = torch.device(device_str)
        model = load(args.model_name, format='huggingface', trust_remote_code=not args.disable_trust_remote_code)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model() # replace vision_tower
            vision_tower.to(device=model.device, dtype=model.dtype)
        image_processor = vision_tower.image_processor
        model = model.to(torch_device)
        model_path = args.model_name
        model_name = get_model_name_from_path(model_path)
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





