import argparse
import copy

parser = argparse.ArgumentParser()
import torch

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.functional import F

from torch.autograd import Function

from datasets import load_from_disk
from torch.utils.data import DataLoader

# import smooth_quant
from evaluation import evaluate as lm_evaluate
import os
from transformers import set_seed
import json
from functools import partial
from torch.amp import autocast

# torch.use_deterministic_algorithms(True)
# os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/models/huggingface"
os.environ['TRANSFORMERS_OFFLINE'] = '0'

parser.add_argument(
    "--model_name", nargs="?", default="/models/opt-125m"
)

parser.add_argument("--group_size", default=128, type=int,
                    help="weight_quantization config")

parser.add_argument("--num_bits", default=4, type=int,
                    help="number of  bits")

parser.add_argument("--cal_grad_batch_size", default=8, type=int,
                    help="cal_grad_batch_size")

parser.add_argument("--batch_size", default=32, type=int,
                    help="batch_size")

parser.add_argument("--cal_grad_fw_bs", default=8, type=int,
                    help="cal_grad_batch_size")

parser.add_argument("--device", default=0, type=str,
                    help="device gpu int number, or 'cpu' ")

parser.add_argument("--sym", action='store_true',
                    help=" sym quantization")
#
# parser.add_argument("--quant_lm_head", action='store_true',
#                     help=" quant lm head")

parser.add_argument("--iters", default=400, type=int,
                    help=" iters")

parser.add_argument("--dynamic_max_gap", default=0, type=int,
                    help=" dynamic max gap")

parser.add_argument("--use_mse", action='store_true',
                    help=" use mse to get best qdq")

parser.add_argument("--use_quant_input", action='store_true',
                    help=" whether use quant_input")

parser.add_argument("--sampler", default="rand", type=str,
                    help="")

parser.add_argument("--clip_val", default=0.5, type=float,
                    help="clip value")

parser.add_argument("--lr", default=0.0025, type=float,
                    help="step size")

parser.add_argument("--lr_decay_type", default="linear", type=str,
                    help="lr decay type")

parser.add_argument("--momentum", default=-1, type=float,
                    help="momentum")

parser.add_argument("--seed", default=42, type=int,
                    help="seed")

parser.add_argument("--eval_fp16", action='store_true',
                    help=" fp32")

parser.add_argument("--amp", action='store_true',
                    help=" amp")

parser.add_argument("--with_attention", action='store_true',
                    help="opt llama with attention")

parser.add_argument("--seq_len", default=512, type=int,
                    help=" seqen lenght")

parser.add_argument("--samples", default=512, type=int,
                    help="samples")

parser.add_argument("--lr_wr", default=0.0, type=float,
                    help="lr warmup ratio")

parser.add_argument("--tasks", default=["lambada_openai", "hellaswag", "winogrande", "piqa"],
                    help=" fp32")

args = parser.parse_args()

print(args.model_name, flush=True)
tasks = args.tasks
set_seed(args.seed)
if args.device == "cpu":
    device_str = "cpu"
else:
    device_str = f"cuda:{int(args.device)}"
cuda_device = torch.device(device_str)

if args.eval_fp16:
    model_name = args.model_name

    if model_name[-1] == "/":
        model_name = model_name[:-1]

    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True,
    )
    model.half()
    model = model.to(cuda_device)
    model_name = args.model_name
    results = lm_evaluate(model="hf-causal",
                          model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=float16',
                          user_model=model, tasks=tasks,
                          device=device_str,
                          batch_size=args.batch_size)
    # datasets = ['wikitext2', 'ptb', 'c4']
    datasets = ['wikitext2', 'ptb-new', 'c4-new']

    from gptq_data_loader import get_loaders


    @torch.no_grad()
    def eval_same_with_gptq(model, testenc, dev):
        print('Evaluating ...', flush=True)
        # model.eval()
        model.to(dev)

        testenc = testenc.input_ids
        nsamples = testenc.numel() // model.seqlen

        use_cache = model.config.use_cache
        model.config.use_cache = False

        testenc = testenc.to(dev)
        nlls = []
        for i in range(nsamples):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc[
                           :, (i * model.seqlen):((i + 1) * model.seqlen)
                           ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print(ppl.item())

        model.config.use_cache = use_cache
        return ppl.item()


    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=args.model_name, seqlen=model.seqlen
        )
        print(dataset, flush=True)
        ppl = eval_same_with_gptq(model, testloader, str(model.device))
        results.update({dataset: ppl})

    exit()


# res = 0
# cum = res
# for i in range(50):
#     res = res*0.5+0.01*i/50
#     cum+=res
#
# print(cum)
# exit()
class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization

    gemmlowp style scale+shift quantization. See more details in
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md.

    We DO NOT recommend affine quantization on weights for performance reason. There might be value to affine quantize
    activation as it can be cancelled by bias and comes with no performance penalty. This functionality is only added
    for experimental purpose.
    """

    @staticmethod
    def forward(ctx, inputs, num_bits=4, group_size=128, schema="asym", grad=None):
        """

        As it will be only applied on activation with per tensor granularity, broadcast is not needed.

        Args:
            ctx: Pytorch convention.
            inputs: A Tensor of type float32.
            min_range: A float.
            max_range: A float.
            num_bits: An integer

        Returns:
            outputs: A Tensor of type output_dtype
        """
        ##ctx.save_for_backward(inputs, min_range, max_range)
        return quant_weight(inputs, num_bits, group_size, schema, grad)[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs

        Returns:
            grad_inputs: A tensor of gradient
        """
        return grad_outputs, None, None, None, None
        # inputs, min_range, max_range = ctx.saved_tensors
        # min_range = min_range.unsqueeze(dim=-1)
        # max_range = max_range.unsqueeze(dim=-1)
        # zero = grad_outputs.new_zeros(1)
        # grad_inputs = torch.where((inputs <= max_range) * (inputs >= min_range), grad_outputs, zero)
        # return grad_inputs, None, None, None


def quant_weight_asym(weight, num_bits=4, grad=None):
    maxq = torch.tensor(2 ** num_bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device)
    wmin = torch.minimum(weight.min(1)[0], zeros)
    wmax = torch.maximum(weight.max(1)[0], zeros)
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = (wmax - wmin) / maxq
    zp = torch.round(-wmin / scale)
    scale.unsqueeze_(dim=-1)
    zp.unsqueeze_(dim=-1)
    if grad != None:
        # grad /= torch.abs(grad)
        # grad *= 0.5
        # grad = torch.clip(grad / scale, -0.5, 0.5)
        int_w = torch.round(weight / scale + grad)
        q = torch.clamp(int_w + zp, 0, maxq)
    else:

        q = torch.clamp(torch.round(weight / scale) + zp, 0, maxq)
    return scale * (q - zp), grad


def quant_weight_sym(weight, num_bits=4):
    # assert num_bits > 1, "symmetric schema only supports num_bits > 1"
    maxq = torch.tensor(2 ** (num_bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-2 ** (num_bits - 1)).to(weight.device)
    if num_bits == 1:
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)

    wmax = torch.abs(weight).max(1)[0]
    tmp = (wmax == 0)
    wmax[tmp] = +1
    scale = wmax / ((maxq - minq) / 2)
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(torch.round(weight / scale), minq, maxq)
    return scale * q


def quant_weight_actor(weight, num_bits, schema, grad):
    assert num_bits > 0, "num_bits should be larger than 0"
    if schema == "sym":
        return quant_weight_sym(weight, num_bits, grad)
    else:
        return quant_weight_asym(weight, num_bits, grad)


def quant_weight(weight, num_bits=4, group_size=-1, schema="asym", grad=None):
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(weight, num_bits, schema=schema, grad=grad)

    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if grad != None:
            grad = grad.reshape(-1, group_size)
        weight, grad = quant_weight_actor(weight, num_bits, schema=schema, grad=grad)

        weight = weight.reshape(orig_shape)
        if grad != None:
            grad = grad.reshape(-1, group_size)
        return weight, grad
    else:
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        if grad != None:
            grad1 = grad[:, :split_index]
            grad1 = grad1.reshape(-1, group_size)
        else:
            grad1 = None
        weight1, grad1 = quant_weight_actor(weight1, num_bits, schema=schema, grad=grad1)
        weight1 = weight1.reshape(orig_shape[0], split_index)
        if grad1 != None:
            grad1 = grad1.reshape(orig_shape[0], split_index)

        weight2 = weight[:, split_index:]
        if grad != None:
            grad2 = grad[:, split_index:]
        weight2, grad2 = quant_weight_actor(weight2, num_bits, schema=schema, grad=grad2)
        weight = torch.cat([weight1, weight2], dim=1)
        if grad != None:
            grad = torch.cat([grad1, grad2], dim=1)
        return weight, grad


class SaveInputs:
    def __init__(self, model, dataloader, seqlen=256):
        self.model = model.eval()
        self.dataloader = dataloader
        # self.op_types = ['Linear']
        self.inputs = {}
        # self.outputs = {}
        self.block_modules = []
        target_m = None
        for n, m in model.named_modules():
            if hasattr(type(m), "__name__") and 'ModuleList' in type(m).__name__:
                target_m = (n, m)

        self.tmps = []
        for n, m in target_m[1].named_children():
            self.tmps.append(target_m[0] + "." + n)
        self.seq_len = seqlen


    @torch.no_grad()
    def get_forward_func(self, name):

        def forward(block, hidden_states, **kwargs):
            if name in self.inputs:
                data = torch.cat([self.inputs[name]['input_ids'], hidden_states.to("cpu")], dim=0)
                self.inputs[name]['input_ids'] = data
            else:
                self.inputs[name] = {}
                self.inputs[name]['input_ids'] = hidden_states.to("cpu")

            if kwargs != None and len(kwargs) > 0:
                if "position_ids" in kwargs.keys() and kwargs["position_ids"] != None:
                    self.inputs[name]["position_ids"] = kwargs["position_ids"].to("cpu")
                # if "attention_mask" in kwargs.keys() and kwargs[
                #     "attention_mask"] != None and "bloom" in args.model_name:
                if "attention_mask" in kwargs.keys() and kwargs[
                    "attention_mask"] != None and (args.with_attention or "bloom" in args.model_name):
                    if "attention_mask" in self.inputs[name] and kwargs["attention_mask"] != None:
                        self.inputs[name]["attention_mask"] = torch.cat(
                            [self.inputs[name]['attention_mask'], kwargs['attention_mask'].to("cpu")], dim=0)
                    else:
                        self.inputs[name]["attention_mask"] = kwargs['attention_mask'].to("cpu")
                if "bloom" in args.model_name and "alibi" in kwargs.keys():
                    alibi = kwargs["alibi"]
                    batch = kwargs['attention_mask'].shape[0]
                    alibi = alibi.reshape(batch, -1, alibi.shape[1], alibi.shape[2])
                    if "alibi" in self.inputs[name].keys():
                        self.inputs[name]["alibi"] = torch.cat(
                            [self.inputs[name]['alibi'], alibi.to("cpu")], dim=0)
                    else:
                        self.inputs[name]["alibi"] = alibi.to("cpu")
            raise NotImplementedError

            # return block.orig_forward(hidden_states, **kwargs)

        return forward

    @torch.no_grad()
    def get_input_outputs(self, n_samples=args.samples):
        if args.amp:
            self.model = self.model.half()
        total_cnt = 0
        # handels = self._add_input_output_observer()
        self._replace_forward()
        for data in self.dataloader:
            if data == None:
                continue

            input_ids = data['input_ids'].to(self.model.device)
            if input_ids.shape[-1] < seqlen:
                continue
            # print(input_ids)
            # attention_mask = data['attention_mask'].to(self.model.device)
            # if args.amp:
            #     with autocast(device_type="cuda"):
            #         self.model(input_ids)
            # else:
            #     self.model(input_ids)
            if total_cnt + input_ids.shape[0] > n_samples:
                input_ids = input_ids[:n_samples - total_cnt, ...]
            try:
                self.model(input_ids)  ##no amp to ease the experiment
            except:
                pass
            total_cnt += input_ids.shape[0]
            if total_cnt >= n_samples:
                break
        self._recover_forward()
        if args.amp:
            self.model = self.model.to(torch.float)

        # for handle in handels:
        #     handle.remove()
        # for key in self.inputs.keys():
        #     data = self.inputs[key]
        #     self.inputs[key] = torch.cat(data, dim=0)
        # for key in self.outputs.keys():
        #     data = self.outputs[key]
        #     self.outputs[key] = torch.cat(data, dim=0)
        # return self.inputs, self.outputs

    def _recover_forward(self):
        for n, m in self.model.named_modules():
            if "lm_head" in n:
                continue
            if n == self.tmps[0]:
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
                break

    def _replace_forward(self):
        for n, m in self.model.named_modules():
            if "lm_head" in n:
                continue
            # if hasattr(type(m), "__name__") and 'ModuleList' in type(m).__name__:
            if n == self.tmps[0]:
                m.orig_forward = m.forward
                m.forward = partial(self.get_forward_func(n), m)
                break



model_name = args.model_name

if model_name[-1] == "/":
    model_name = model_name[:-1]

model = AutoModelForCausalLM.from_pretrained(
    model_name, low_cpu_mem_usage=True,
)
model = model.to(cuda_device)
if "llama" in model_name:
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)


@torch.no_grad()
def q_dq_weight(model: torch.nn.Module, num_bits=4, group_size=128, schema='asym'):
    for n, m in model.named_modules():
        # if args.quant_lm_head:
        #     if isinstance(m, torch.nn.Linear):
        #         m.weight.data.copy_(
        #             quant_weight(m.weight, num_bits=num_bits, group_size=group_size, schema=schema))
        # else:
        if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
            m.weight.data.copy_(
                quant_weight(m.weight, num_bits=num_bits, group_size=group_size, schema=schema)[0])


if args.iters <= 0:
    q_dq_weight(model, num_bits=args.num_bits, group_size=args.group_size)
    model.half()
    model = model.to(cuda_device)
    model_name = args.model_name
    results = lm_evaluate(model="hf-causal",
                          model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=float16',
                          user_model=model, tasks=tasks,
                          device=device_str,
                          batch_size=args.batch_size)
    datasets = ['wikitext2', 'ptb-new', 'c4-new']

    from gptq_data_loader import get_loaders


    @torch.no_grad()
    def eval_same_with_gptq(model, testenc, dev):
        print('Evaluating ...', flush=True)
        # model.eval()
        model.to(dev)

        testenc = testenc.input_ids
        nsamples = testenc.numel() // model.seqlen

        use_cache = model.config.use_cache
        model.config.use_cache = False

        testenc = testenc.to(dev)
        nlls = []
        for i in range(nsamples):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc[
                           :, (i * model.seqlen):((i + 1) * model.seqlen)
                           ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print(ppl.item())

        model.config.use_cache = use_cache
        return ppl.item()


    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=args.model_name, seqlen=model.seqlen
        )
        print(dataset, flush=True)
        ppl = eval_same_with_gptq(model, testloader, str(model.device))
        results.update({dataset: ppl})

    exit()


dataset_name = "NeelNanda/pile-10k"
# calib_dataset = load_dataset(dataset_name, split="train")
# # calib_dataset.save_to_disk("pile_10k")
if os.path.exists(dataset_name.split('/')[-1]):
    calib_dataset = load_from_disk(dataset_name.split('/')[-1])
else:
    calib_dataset = load_dataset(dataset_name, split="train")
    calib_dataset.save_to_disk(dataset_name.split('/')[-1])


if "opt" in model_name:
    seqlen = model.config.max_position_embeddings
    model.seqlen = model.config.max_position_embeddings
else:
    seqlen = 2048
    model.seqlen = seqlen

seqlen = args.seq_len


def tokenize_function(examples):
    example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
    # example = tokenizer(examples["text"], return_tensors='pt', padding=True)
    # example = tokenizer(examples["text"])
    return example


@torch.no_grad()
def collate_batch(batch):
    from torch.nn.functional import pad
    input_ids_new = []
    for text in batch:
        input_ids = text["input_ids"]
        if input_ids.shape[0] < seqlen:
            continue
        input_ids = input_ids[:seqlen]
        input_ids_list = input_ids.tolist()
        if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
            continue
        input_ids_new.append(input_ids)
    if len(input_ids_new) == 0:
        return None
    tmp = torch.vstack(input_ids_new)
    res = {}
    res["input_ids"] = tmp
    return res


calib_dataset = calib_dataset.shuffle(seed=args.seed)
calib_dataset = calib_dataset.map(tokenize_function, batched=True)
# calib_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
calib_dataset.set_format(type='torch', columns=['input_ids'])
# if "llama" in args.model_name:
#     args.batch_size =1
calib_dataloader = DataLoader(
    calib_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

model = model.eval()
save_input_file = f"{(args.model_name).split('/')[-1]}_input_block.pt"


import time

if "opt" in model_name:
    seqlen = model.config.max_position_embeddings
    model.seqlen = model.config.max_position_embeddings
else:
    seqlen = 2048
    model.seqlen = seqlen
seqlen = args.seq_len
start_time = time.time()
save_input_actor = SaveInputs(model, calib_dataloader, seqlen)
save_input_actor.get_input_outputs()
input_info = save_input_actor.inputs


def get_module(model, key):
    """Get module from model by key name

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    # print(key, flush=True)
    attrs = key.split('.')
    module = model
    for attr in attrs:
        try:
            attr = int(attr)
            module = module[attr]
        except:
            module = getattr(module, attr)
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    attrs = key.split('.')
    module = model
    for attr in attrs[:-1]:
        try:
            attr = int(attr)
            module = module[attr]
        except:
            module = getattr(module, attr)
    setattr(module, attrs[-1], new_module)


#
# def quant_weight_block(module, num_bits, group_size, schema, grads, block_name):
#     for n, m in module.named_modules():
#         if isinstance(m, torch.nn.Linear):
#             grad = None
#             if grads != None:
#                 grad = grad[block_name + "." + n]
#             qdq_weight, _ = quant_weight(m.weight, num_bits, group_size, schema, grad=grad)
#             m.weight.data.copy_(q_dq_weight)


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, num_bits, group_size, schema, grad=None):
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.tensor_quant = FakeAffineTensorQuantFunction().apply
        self.num_bits = num_bits
        self.group_size = group_size
        self.schema = schema
        self.grad = grad

    def update_grad(self, grad):
        self.grad = grad

    def forward(self, x):
        weight = self.orig_layer.weight
        weight_q = FakeAffineTensorQuantFunction().apply(weight, self.num_bits, self.group_size, self.schema,
                                                         self.grad)
        return F.linear(x, weight_q, self.orig_layer.bias)


def wrapper_block(block, num_bits, group_size, schema):
    for n, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            new_m = WrapperLinear(m, num_bits, group_size, schema)
            set_module(block, n, new_m)


def unwrapper_block(block, num_bits, group_size, schema, grads):
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            orig_layer = m.orig_layer
            grad = None
            if grads != None:
                grad = grads[n]
            q_dq_weight, _ = quant_weight(orig_layer.weight, num_bits, group_size, schema, grad)
            orig_layer.weight.data.copy_(q_dq_weight)
            set_module(block, n, orig_layer)


def collect_grad_and_zero(block):
    grads = {}
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            grad = -m.orig_layer.weight.grad  ##FIXME
            # mean_val = torch.mean(torch.abs(grad))
            # std_val = torch.std(torch.abs(grad))
            # threshold1 = mean_val/100
            # threshold2 = mean_val-3*std_val
            # print(threshold1, threshold2,flush=True)
            # grad[torch.abs(grad) < mean_val/100] = 0##stable the tuning
            grads[n] = copy.deepcopy(grad)
            m.orig_layer.weight.grad.zero_()
    return grads


def get_lr(step, total_steps, lr=0.01, warmup_step=0, lr_type="linear"):
    if warmup_step > 0 and step < warmup_step:
        return (lr - 0.01 * lr) * float(step) / warmup_step + 0.01 * lr
        ##return lr * float(step) / warmup_step

    if lr_type == "const":
        current_lr = args.lr
    elif lr_type == "linear":
        current_lr = args.lr - float(step - warmup_step) / (total_steps - warmup_step) * lr
    elif lr_type == "cos":
        import math
        current_lr = math.cos(float(step - warmup_step) / (total_steps - warmup_step) * math.pi / 2) * lr
    else:
        raise NotImplemented
    return current_lr



def quant_block(block, input_ids, input_others, output, num_bits, group_size, schema, q_input=None):
    # block.train()
    best_loss = torch.finfo(torch.float).max
    grad = None
    if args.momentum > 0:
        grad_m = None
    # module_copy = copy.deepcopy(block)

    # input = input.to(cuda_device)
    ##output = output.to(cuda_device)
    # input_ids = input['input_ids']
    # input.pop('input_ids', None)
    # input_others = input
    input = input_ids
    input = input.to(cuda_device)
    # with torch.no_grad():
    #     if "bloom" in args.model_name:
    #         attention_mask = input_others["attention_mask"]
    #         alibi = input_others["alibi"]
    #         alibi_tmp = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
    #         if args.amp:
    #             with autocast(device_type="cuda"):
    #                 output = block(input, attention_mask=attention_mask, alibi=alibi_tmp)[0] * 1000
    #         else:
    #             output = block(input, attention_mask=attention_mask, alibi=alibi_tmp)[0]
    #     else:
    #         if args.amp:
    #             with autocast(device_type="cuda"):
    #                 output = block(input, **input_others)[0]
    #         else:
    #             output = block(input, **input_others)[0]
    output = []
    with torch.no_grad():
        current_bs = args.cal_grad_fw_bs
        for i in range(0, args.samples, current_bs):
            indices = torch.arange(i, i + current_bs).to(torch.long)
            current_input_other = {}
            if "position_ids" in input_others.keys():
                current_input_other["position_ids"] = input_others["position_ids"]
            if len(input.shape) == 3:
                current_input = input[indices, :, :]
                # current_output = output[indices, :, :]
            else:
                n_samples = input.shape[0] // seqlen
                current_input = input.view(n_samples, seqlen, -1)
                # indices = torch.randperm(n_samples)[:pick_samples]
                current_input = current_input[indices, :, :]
                current_input = current_input.view(-1, input.shape[-1])
                # current_output = output.view(n_samples, seqlen, -1)
                # current_output = current_output[indices, :, :]
                # current_output = current_output.view(-1, current_output.shape[-1])
            if "attention_mask" in input_others:
                current_input_other["attention_mask"] = input_others["attention_mask"][indices, ...]
            if "bloom" in args.model_name:
                current_input_other["alibi"] = input_others["alibi"][indices, ...]

            if "bloom" in args.model_name:
                attention_mask = current_input_other["attention_mask"]
                alibi = current_input_other["alibi"]
                alibi_tmp = alibi.view(-1, alibi.shape[2], alibi.shape[3])
                if args.amp:
                    with autocast(device_type="cuda"):
                        tmp_output = block(current_input, attention_mask=attention_mask, alibi=alibi_tmp)[0]
                else:
                    tmp_output = block(current_input, attention_mask=attention_mask, alibi=alibi_tmp)[0]
            else:
                if args.amp:
                    with autocast(device_type="cuda"):
                        tmp_output = block.forward(current_input, **current_input_other)[0]
                else:
                    tmp_output = block.forward(current_input, **current_input_other)[0]
            output.append(tmp_output)
        output = torch.cat(output, dim=0)

    # with torch.no_grad():
    #     if "bloom" in args.model_name:
    #         attention_mask = input_others["attention_mask"]
    #         alibi = input_others["alibi"]
    #         alibi_tmp = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
    #         if args.amp:
    #             with autocast(device_type="cuda"):
    #                 output = block(input, attention_mask=attention_mask, alibi=alibi_tmp)[0]
    #         else:
    #             output = block(input, attention_mask=attention_mask, alibi=alibi_tmp)[0]
    #     else:
    #         if args.amp:
    #             with autocast(device_type="cuda"):
    #                 output = block(input, **input_others)[0]
    #         else:
    #             output = block(input, **input_others)[0]

    # print("input.shape", input.shape)
    # print("output.shape", output.shape)
    # best_module = copy.deepcopy(block.cpu())
    if q_input != None:
        input = q_input.to(cuda_device)

    wrapper_block(block, num_bits, group_size, schema)
    search_iters = args.iters
    best_grad = None
    pick_samples = args.cal_grad_batch_size
    if len(input.shape) == 3:
        n_samples = input.shape[0]
    else:
        n_samples = input.shape[0] // seqlen
    if args.sampler != "rand":
        indices = torch.randperm(n_samples)[:pick_samples]
    last_best_iter = 0
    for i in range(search_iters):
        current_input = input
        current_output = output
        current_input_other = copy.deepcopy(input_others)

        if args.cal_grad_batch_size > 0:
            if args.sampler == "rand":
                indices = torch.randperm(n_samples)[:pick_samples]

            if len(input.shape) == 3:

                current_input = input[indices, :, :]
                current_output = output[indices, :, :]
            else:
                n_samples = input.shape[0] // seqlen
                current_input = current_input.view(n_samples, seqlen, -1)
                # indices = torch.randperm(n_samples)[:pick_samples]
                current_input = current_input[indices, :, :]
                current_input = current_input.view(-1, input.shape[-1])
                current_output = current_output.view(n_samples, seqlen, -1)
                current_output = current_output[indices, :, :]
                current_output = current_output.view(-1, current_output.shape[-1])
            if "attention_mask" in current_input_other:
                current_input_other["attention_mask"] = input_others["attention_mask"][indices, ...]
            if "bloom" in args.model_name:
                current_input_other["alibi"] = input_others["alibi"][indices, ...]

            # if len(input.shape) == 3:
            #     n_samples = input.shape[0]
            #     ##indices = torch.randperm(n_samples)[:pick_samples]
            #     current_input = input[:pick_samples, :, :]
            #     # current_output = output[indices, :, :]
            # else:
            #     n_samples = input.shape[0] // seqlen
            #     current_input = current_input.view(n_samples, seqlen, -1)
            #     ##indices = torch.randperm(n_samples)[:pick_samples]
            #     current_input = current_input[:pick_samples, :, :]
            #     current_input = current_input.view(-1, input.shape[-1])
            #
            #     # current_output = current_output.view(n_samples, seqlen, -1)
            #     # current_output = current_output[indices, :, :]
            #     # current_output = current_output.view(-1, current_output.shape[-1])

        # quant_weight_block(block, num_bits, group_size, schema, grads=grad, block_name=block_name)
        start_index = 0
        step_size = args.cal_grad_fw_bs
        end_index = 0
        total_loss = 0
        mse_loss = torch.nn.MSELoss()
        attention_mask = current_input_other["attention_mask"]
        while 1:
            end_index += step_size
            end_index = min(end_index, current_input.shape[0])
            tmp_input = current_input[start_index:end_index, ...]

            current_input_other.pop('attention_mask', None)
            attention_mask_tmp = attention_mask[start_index:end_index,...]
            if "bloom" in args.model_name:
                alibi = current_input_other["alibi"][start_index:end_index,...]
                alibi_tmp = alibi.view(-1, alibi.shape[2], alibi.shape[3])
                if args.amp:
                    with autocast(device_type="cuda"):
                        output_q = block(tmp_input, attention_mask=attention_mask_tmp, alibi=alibi_tmp)
                else:
                    output_q = block(tmp_input, attention_mask=attention_mask_tmp, alibi=alibi_tmp)
            else:

                if args.amp:
                    with autocast(device_type="cuda"):
                        output_q = block.forward(tmp_input,attention_mask = attention_mask_tmp, **current_input_other)
                else:
                    output_q = block.forward(tmp_input, attention_mask = attention_mask_tmp, **current_input_other)
            if isinstance(output_q, list) or isinstance(output_q, tuple):
                output_q = output_q[0]
            # gap = (current_output[start_index:end_index] - output_q)
            if args.amp:
                with autocast(device_type="cuda"):
                    loss = mse_loss(output_q, current_output[start_index:end_index]) * 1000
            else:
                loss = mse_loss(output_q, current_output[start_index:end_index])
            total_loss += loss.item()
            loss.backward()
            if end_index == current_input.shape[0]:
                break
            start_index = end_index

        # # if i != 0:
        if total_loss < best_loss:
            best_loss = total_loss
            if args.use_mse:
                # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)
                best_grad = copy.deepcopy(grad)
                last_best_iter = i
        if not args.use_mse:
            best_grad = grad
        if args.use_mse:
            if args.dynamic_max_gap > 0 and i - last_best_iter >= args.dynamic_max_gap:
                break
        new_grad = collect_grad_and_zero(block)

        warmup_step = int(args.iters * args.lr_wr)
        current_lr = get_lr(i, args.iters, args.lr, warmup_step, args.lr_decay_type)
        # if args.lr_decay_type == "const":
        #     current_lr = args.lr
        # elif args.lr_decay_type == "linear":
        #     current_lr = args.lr - float(i) / args.iters * args.lr
        # elif args.lr_decay_type == "cos":
        #     import math
        #     current_lr = math.cos(float(i) / args.iters*math.pi/2) * args.lr
        # else:
        #     raise NotImplemented

        for key in new_grad.keys():
            new_grad[key] = torch.sign(new_grad[key]) * current_lr

            # new_grad[key] = torch.sign(new_grad[key]) * (args.lr)

        if grad == None:
            grad = new_grad
            grad_m = new_grad
        else:
            for key in new_grad.keys():
                if args.momentum > 0:
                    grad_m[key] = args.momentum * grad_m[key] + new_grad[key]
                    grad[key] += grad_m[key]
                else:
                    grad[key] += new_grad[key]

        clip_value = args.clip_val

        for key in grad.keys():
            grad[key] = torch.clip(grad[key], -clip_value, clip_value)
        for n, m in block.named_modules():
            if isinstance(m, WrapperLinear):
                m.update_grad(grad[n])

    # for n, m in block.named_modules():
    #     if isinstance(m, WrapperLinear):
    #         for tmp in range(0, 101):
    #             target = float(tmp) / 100-0.5
    #             print(target, torch.sum(torch.abs(best_grad[n]- target)<1e-6) / best_grad[n].numel(), flush=True)
    # for n, m in block.named_modules():
    #     if isinstance(m, WrapperLinear):
    #         for tmp in range(0, 11):
    #             target = float(tmp) / 10-0.5
    #             if best_grad[n]==None:
    #                 print(n)
    #                 continue
    #             print(target, torch.sum(torch.abs(best_grad[n]- target)<0.1) / best_grad[n].numel(), flush=True)
    unwrapper_block(block, num_bits, group_size, schema, best_grad)
    block.eval()
    if args.use_quant_input:
        with torch.no_grad():
            if "bloom" in args.model_name:
                attention_mask = input_others["attention_mask"]
                alibi = input_others["alibi"]
                alibi_tmp = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
                if args.amp:
                    with autocast(device_type="cuda"):
                        q_output = block(input, attention_mask=attention_mask, alibi=alibi_tmp)[0]
                else:
                    q_output = block(input, attention_mask=attention_mask, alibi=alibi_tmp)[0]
            else:
                q_output = block(input, **input_others)[0]
        input = input.to("cpu")

        return q_output, output

    else:
        input = input.to("cpu")
        return None, output
    ##module.weight.copy_(best_q_dq_weight)


# @torch.no_grad()
def q_dq_weight_round(model: torch.nn.Module, num_bits=4, group_size=128, schema='asym'):
    q_input = None
    torch.cuda.empty_cache()
    input_others = None
    ##set_seed(args.seed)  ##to reduce randomness of script run
    for n in save_input_actor.tmps:

        if "lm_head" in n:
            continue
        m = get_module(model, n)

        print(n, flush=True)
        m = m.to(cuda_device)
        m.eval()
        input = None
        q_input = None
        if n in input_info.keys():
            input = input_info[n]
            input_ids = input['input_ids']
            input.pop('input_ids', None)
            input_others = input
            for key in input_others.keys():
                input_others[key] = input_others[key].to(cuda_device)

        q_input, input_ids = quant_block(m, input_ids, input_others, None, num_bits=num_bits, group_size=group_size,
                                         schema=schema,
                                         q_input=q_input)
        m.eval()
        m = m.to("cpu")
        torch.cuda.empty_cache()
    for key in input_others.keys():
        input_others[key] = input_others[key].to("cpu")
    torch.cuda.empty_cache()


model.eval()
if args.iters <= 0:
    q_dq_weight(model, num_bits=args.num_bits, group_size=args.group_size)
else:
    model = model.to("cpu")

    q_dq_weight_round(model, num_bits=args.num_bits, group_size=args.group_size)
    end_time = time.time()
    print(end_time - start_time, flush=True)
# q_dq_weight(model, num_bits=args.num_bits, group_size=args.group_size)
torch.cuda.empty_cache()
# # torch.save(model, model_name.split('/')[-1]+"_seq2048_mse_samplers512_iter400.pt")
# output_dir =  model_name.split('/')[-1]+"_seq2048_mse_samplers512_calbs8_iter400"
#
# if output_dir is not None:
#
#     model.save_pretrained(output_dir)
#
#     tokenizer.save_pretrained(output_dir)

model = model.half()

model = model.to(cuda_device)

model.eval()
print(args.model_name, flush=True)
results = lm_evaluate(model="hf-causal",
                      model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=float16',
                      user_model=model, tasks=tasks,
                      device=str(model.device),
                      batch_size=32)

# datasets = ['wikitext2', 'ptb', 'c4']
datasets = ['wikitext2', 'ptb-new', 'c4-new']

from gptq_data_loader import get_loaders


@torch.no_grad()
def eval_same_with_gptq(model, testenc, dev):
    print('Evaluating ...', flush=True)
    # model.eval()
    model.to(dev)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()


for dataset in datasets:
    dataloader, testloader = get_loaders(
        dataset, seed=0, model=args.model_name, seqlen=model.seqlen
    )
    print(dataset, flush=True)
    ppl = eval_same_with_gptq(model, testloader, str(model.device))
    results.update({dataset: ppl})
