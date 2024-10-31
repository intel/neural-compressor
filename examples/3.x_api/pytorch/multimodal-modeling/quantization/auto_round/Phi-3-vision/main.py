import argparse
# import sys
# sys.path.insert(0, '../../..')
parser = argparse.ArgumentParser()
import torch
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.use_deterministic_algorithms(True, warn_only=True)
import copy
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, Optional, List, Union, Sequence
import transformers
from model.processing_phi3_v import Phi3VProcessor
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import subprocess
LLaVA_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN = "<|image_1|>"
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
from neural_compressor.torch.utils.utility import (get_multimodal_block_names,
                                                    get_layer_names_in_block,
                                                    detect_device,
                                                    run_fn_for_vlm_autoround
                                                    )
from neural_compressor.torch.quantization import (AutoRoundConfig,
                                                    prepare,
                                                    convert,
                                                    load)
@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    
    
def llava_to_openai(data):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for entry in data:
        transformed_entry = {
            "role": role_mapping.get(entry["from"], entry["from"]),
            "content": entry["value"].replace(LLaVA_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN),
        }
        transformed_data.append(transformed_entry)

    return transformed_data


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: Union[int, str],
        processor: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_seq_length = data_args.max_seq_length

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        processor = self.processor
        if "image" in sources[0]:
            image_file = os.path.basename(self.list_data_dict[i]["image"])
            image_folder = self.data_args.image_folder
            image_file = os.path.join(image_folder, image_file)
            image = [Image.open(image_file).convert("RGB")]
        else:
            image = None
        sources = copy.deepcopy([e["conversations"] for e in sources])
        for i in range(len(sources)):
            sources[i] = llava_to_openai(sources[i])

        prompt = processor.tokenizer.apply_chat_template(
            sources[0], tokenize=False, add_generation_prompt=True
        )
        data_dict = processor(prompt, image, return_tensors="pt")

        if self.padding:
            training_length = self.max_seq_length
            if 'pixel_values' not in data_dict:
                data_dict['pixel_values'] = torch.zeros([1, 17, 3, 336, 336], dtype=torch.bfloat16)
                data_dict['image_sizes'] = torch.zeros([1, 2], dtype=torch.int64)
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                attention_mask=data_dict["attention_mask"][0],
                pixel_values=data_dict["pixel_values"][0],
                image_sizes=data_dict["image_sizes"][0],
                labels=data_dict["labels"][0],
            )
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        pixel_values = [instance["pixel_values"] for instance in instances]
        pixel_values = torch.stack(pixel_values, dim=0)
        image_sizes = [instance["image_sizes"] for instance in instances]
        image_sizes = torch.stack(image_sizes, dim=0)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch


def create_data_loader(dataset, batch_size=1, data_collator=None):
    assert batch_size == 1, "batch_size must be 1"
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    return data_loader


if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="microsoft/Phi-3-vision-128k-instruct")
    
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

    parser.add_argument("--seqlen", default=2048, type=int,
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
    
    parser.add_argument("--model_max_length", default=2048, type=int,
                        help="")
    
    parser.add_argument("--act_bits", default=32, type=int,
                    help="activation bits")
    
    parser.add_argument("--quant_vision", action='store_true',
                        help="To determine whether the quantization should handle vision component.")
    
    parser.add_argument("--enable_safe_serialization", action='store_true',
                        help="To determine whether the save_pretrained process should use safe_serialization.")
    
    # ========== Calibration Datasets ============= 
    parser.add_argument("--image_folder", default="coco", type=str,
                        help="The dataset for quantization training. It can be a custom one.")
    
    parser.add_argument("--question_file", default=None, type=str,
                            help="The dataset for quantization training. It can be a custom one.")
    
    # ================= Evaluation Related =====================
    parser.add_argument("--tasks", #wikitext
                        default="lambada_openai,hellaswag,winogrande,piqa,mmlu,truthfulqa_mc1," \
                                "truthfulqa_mc2,openbookqa,boolq,rte,arc_easy,arc_challenge",
                        help="lm-eval tasks for lm_eval version 0.4")
    
    parser.add_argument("--eval-dataset", type=str, default="textvqa_val")

    args = parser.parse_args()

    if args.quantize:
        tasks = args.tasks
            
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
        torch_device = torch.device(device_str)
        
        torch.manual_seed(1234)
        model_name = args.model_name
        seqlen = args.seqlen
        questions = json.load(open(args.question_file, "r"))
        torch_dtype = "auto"
        if "hpu" in device_str:
            torch_dtype = torch.bfloat16
        if args.model_dtype != None:
            if args.model_dtype == "float16" or args.model_dtype == "fp16":
                torch_dtype = torch.float16
            if args.model_dtype == "bfloat16" or args.model_dtype == "bfp16":
                torch_dtype = torch.bfloat16
                
        # config = AutoConfig.from_pretrained(
        #     model_name,
        #     trust_remote_code=not args.disable_trust_remote_code,
        # )
        # config.use_cache = False
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=not args.disable_trust_remote_code,
            _attn_implementation='eager' # _attn_implementation='flash_attention_2' to enable flash attention
        )
        seqlen = args.seqlen
        processor = Phi3VProcessor.from_pretrained(model_name)
        tokenizer = processor.tokenizer
        data_args = DataArguments(
            data_path=args.question_file,
            is_multimodal=True,
            image_folder=args.image_folder,
            max_seq_length=seqlen,
        )
        dataset = LazySupervisedDataset(
            data_path=args.question_file, processor=processor, data_args=data_args
        )
        data_collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
        dataloader = create_data_loader(dataset, batch_size=args.train_bs, data_collator=data_collator)
            
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
        quant_config.set_local("model.vision_embed_tokens.img_projection*", AutoRoundConfig(dtype="fp32"))
            
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

        from neural_compressor.torch.utils import (LoadFormat,)
        user_model.save(args.output_dir, format=LoadFormat.HUGGINGFACE, safe_serialization=False)
        if tokenizer is not None:
            tokenizer.save_pretrained(args.output_dir)
        if processor is not None:
            processor.save_pretrained(args.output_dir)

    
    # if args.accuracy:
    #     from eval.evaluation import simple_evaluate
    #     device_str = detect_device(args.device)
    #     tasks = args.tasks
    #     if isinstance(tasks, str):
    #         tasks = tasks.split(',')
    #     model_args = f"pretrained={args.model_name}"
    #     model_args = model_args + f",trust_remote_code={not args.disable_trust_remote_code}"
    #     model_args += f",autogptq=True,gptq_use_triton=True"
    #     user_model = load(args.model_name, format='huggingface', \
    #                         trust_remote_code=not args.disable_trust_remote_code, _attn_implementation='eager')
    #     if args.act_bits <= 8:
    #         user_model = model.to(device_str)

    #     res = simple_evaluate(model="hf", model_args=model_args,
    #                         tasks=tasks,
    #                         batch_size=args.eval_bs, user_model=user_model)
    #     from lm_eval.utils import make_table
    #     print(make_table(res))



