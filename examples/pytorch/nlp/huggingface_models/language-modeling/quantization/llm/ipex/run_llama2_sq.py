import argparse

from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader

import intel_extension_for_pytorch as ipex

parser = argparse.ArgumentParser('LLaMA generation script (int8 path)', add_help=False)

parser.add_argument(
    "-m", "--model-id", default=None, type=str, required=True, help="your llama model"
)
parser.add_argument(
    "--sq-recipes", default=None, type=str, required=True, help="llama2-7b or llama2-13b"
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k")
parser.add_argument("--output-dir", nargs="?", default="./saved_results")

parser.add_argument(
    "--int8-bf16-mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--padding", action="store_true", help="whether do padding in calib_dataloader")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--alpha", default=0.8, type=float, help="alpha value for smoothquant")
parser.add_argument("--greedy", action="store_true")

args = parser.parse_args()

try:
    ipex._C.disable_jit_linear_repack()
except Exception:
    pass

# amp autocast
if args.int8_bf16_mixed:
    amp_enabled = True
    amp_dtype = torch.bfloat16
else:
    amp_enabled = False
    amp_dtype = torch.float32

num_beams = 1 if args.greedy else 4

# load model
config = AutoConfig.from_pretrained(args.model_id, torchscript=True)
if not hasattr(config, "text_max_length") and args.prompt is None:
    config.text_max_length = int(args.input_tokens) + int(args.max_new_tokens)

user_model = LlamaForCausalLM.from_pretrained(
    args.model_id, config=config, low_cpu_mem_usage=True, torch_dtype=torch.float
)

tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
print("Data type of the model:", user_model.dtype)

# dummy past key value
beam_idx_tmp = torch.zeros(
    (2048, int(args.batch_size * num_beams)), dtype=torch.long
).contiguous()
global_past_key_value = [
    (
        torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
        torch.zeros(
            [
                1,
                user_model.config.num_attention_heads,
                1,
                int(
                    user_model.config.hidden_size
                    / user_model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        torch.zeros(
            [
                1,
                user_model.config.num_attention_heads,
                1,
                int(
                    user_model.config.hidden_size
                    / user_model.config.num_attention_heads
                ),
            ]
        ).contiguous(),
        beam_idx_tmp,
    )
    for i in range(user_model.config.num_hidden_layers)
]


class Evaluator:

    def __init__(self, dataset, tokenizer, batch_size=1, pad_val=1, pad_max=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        if "prompt" in examples:
            example = self.tokenizer(examples["prompt"])
        elif "text" in examples:
            example = self.tokenizer(examples["text"])
        elif "code" in examples:
            example = self.tokenizer(examples["code"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for text in batch:
            input_ids = text["input_ids"]
            if not args.padding:
                input_ids = (
                    input_ids[: int(self.pad_max)]
                    if len(input_ids) > int(self.pad_max)
                    else input_ids
                ) #no_padding
            else:
                pad_len = self.pad_max - input_ids.shape[0] 
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids_padded.append(input_ids)
            attention_mask_padded.append(attention_mask)
            position_ids_padded.append(position_ids)
        return (
            (
                torch.vstack(input_ids_padded),
                torch.vstack(attention_mask_padded),
                torch.vstack(position_ids_padded),
                tuple(global_past_key_value),
            ),
            torch.tensor(last_ind),
        )


calib_dataset = load_dataset(args.dataset, split="train")
user_model.eval()
if args.sq_recipes == "llama2-7b":
    pad_max = 2048
elif args.sq_recipes == "llama2-13b":
    pad_max = 1024
else:
    pad_max = 512
calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=pad_max)
calib_dataloader = DataLoader(
    calib_evaluator.dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=calib_evaluator.collate_batch,
)


def calib_func(prepared_model):
    for i, (
            (input_ids, attention_mask, position_ids, past_key_values),
            last_ind,
    ) in enumerate(calib_dataloader):
        if i == 512:
            break
        prepared_model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )


example_inputs = None
for i, (
        (input_ids, attention_mask, position_ids, past_key_values),
        last_ind,
) in enumerate(calib_dataloader):
    example_inputs = (input_ids, attention_mask, position_ids, past_key_values)
    break

qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=args.alpha)
user_model = ipex.optimize_transformers(
    user_model.eval(),
    dtype=amp_dtype,
    quantization_config=qconfig,
    inplace=True,
    deployment_mode=False,
)

# steps for SmoothQuant with Intel® Neural Compressor
from neural_compressor import PostTrainingQuantConfig, quantization

# quantization recipes
excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
op_type_dict = {"add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}}
recipes = {}
if args.sq_recipes == "llama2-7b":
    recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': 'auto', 'folding': False, 'default_alpha': 0.8,
                                                           'auto_alpha_args': {"alpha_min": 0.8, "alpha_max": 0.99,
                                                                               "alpha_step": 0.01,
                                                                               "shared_criterion": "mean"}}}
elif args.sq_recipes == "llama2-13b":
    recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': 'auto', 'folding': False, 'default_alpha': 0.8,
                                                        'auto_alpha_args': {"alpha_min": 0.75, "alpha_max": 0.99,
                                                                            "alpha_step": 0.01,
                                                                            "shared_criterion": "max"}}}


conf = PostTrainingQuantConfig(
    backend="ipex",
    excluded_precisions=excluded_precisions,
    op_type_dict=op_type_dict,
    recipes=recipes,
    example_inputs=example_inputs,
)
q_model = quantization.fit(
    user_model,
    conf,
    calib_dataloader=calib_dataloader,
    calib_func=calib_func,
)
q_model.save(args.output_dir)
