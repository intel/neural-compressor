import argparse
import copy

parser = argparse.ArgumentParser()
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.functional import F

from torch.autograd import Function

from datasets import load_from_disk
from torch.utils.data import DataLoader

import os
from transformers import set_seed
from functools import partial
from torch.amp import autocast
from eval import eval_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# os.environ["HF_HOME"] = "/models/huggingface"
# os.environ['TRANSFORMERS_OFFLINE'] = '0'


class FakeAffineTensorQuantFunction(Function):
    @staticmethod
    def forward(ctx, inputs, num_bits=4, group_size=128, schema="asym", grad=0):
        return quant_weight(inputs, num_bits, group_size, schema, grad)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None, None, None


def quant_weight_asym(weight, num_bits=4, grad=0):
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
    int_w = torch.round(weight / scale + grad)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp)


def quant_weight_sym(weight, num_bits=4, grad=0):
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
    q = torch.clamp(torch.round(weight / scale + grad), minq, maxq)
    return scale * q


def quant_weight_actor(weight, num_bits, schema, grad):
    assert num_bits > 0, "num_bits should be larger than 0"
    if schema == "sym":
        return quant_weight_sym(weight, num_bits, grad)
    else:
        return quant_weight_asym(weight, num_bits, grad)


def quant_weight(weight, num_bits=4, group_size=-1, schema="asym", grad=0):
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(weight, num_bits, schema=schema, grad=grad)

    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if isinstance(grad, torch.Tensor):
            grad = grad.reshape(-1, group_size)
        weight = quant_weight_actor(weight, num_bits, schema=schema, grad=grad)

        weight = weight.reshape(orig_shape)

        return weight
    else:
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        if isinstance(grad, torch.Tensor):
            grad1 = grad[:, :split_index]
            grad1 = grad1.reshape(-1, group_size)
            grad2 = grad[:, split_index:]
        else:
            grad1 = 0
            grad2 = 0
        weight1 = quant_weight_actor(weight1, num_bits, schema=schema, grad=grad1)
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2 = quant_weight_actor(weight2, num_bits, schema=schema, grad=grad2)
        weight = torch.cat([weight1, weight2], dim=1)

        return weight


class SaveInputs:
    def __init__(self, model, dataloader, seqlen=256, block_name=None):
        self.model = model.eval()
        self.dataloader = dataloader
        self.inputs = {}
        self.block_name = block_name
        self.seqlen = seqlen

    @torch.no_grad()
    def get_forward_func(self, name):

        def forward(block, hidden_states, **kwargs):  ##This may have bug for other models
            if name in self.inputs:
                data = torch.cat([self.inputs[name]['input_ids'], hidden_states.to("cpu")], dim=0)
                self.inputs[name]['input_ids'] = data
            else:
                self.inputs[name] = {}
                self.inputs[name]['input_ids'] = hidden_states.to("cpu")

            if kwargs is not None and len(kwargs) > 0:
                if "position_ids" in kwargs.keys() and kwargs["position_ids"] is not None:
                    self.inputs[name]["position_ids"] = kwargs["position_ids"].to("cpu")
                if "attention_mask" in kwargs.keys() and kwargs["attention_mask"] is not None and (
                        (not args.not_with_attention) or "bloom" in args.model_name):
                    if "attention_mask" in self.inputs[name] and kwargs["attention_mask"] is not None:
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

        return forward

    @torch.no_grad()
    def get_inputs(self, n_samples=512):
        total_cnt = 0
        self._replace_forward()
        for data in self.dataloader:
            if data is None:
                continue

            input_ids = data['input_ids'].to(self.model.device)
            if input_ids.shape[-1] < seqlen:
                continue
            if total_cnt + input_ids.shape[0] > n_samples:
                input_ids = input_ids[:n_samples - total_cnt, ...]
            try:
                self.model(input_ids)
            except:
                pass
            total_cnt += input_ids.shape[0]
            if total_cnt >= n_samples:
                break
        self._recover_forward()
        return self.inputs[self.block_name]

    def _recover_forward(self):
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
                break

    def _replace_forward(self):
        for n, m in self.model.named_modules():
            if n == self.block_name:
                m.orig_forward = m.forward
                m.forward = partial(self.get_forward_func(n), m)
                break


@torch.no_grad()
def q_dq_weight(model: torch.nn.Module, num_bits=4, group_size=128, schema='asym'):
    target_m = None
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and 'ModuleList' in type(m).__name__:
            target_m = (n, m)
    block_names = []
    for n, m in target_m[1].named_children():
        block_names.append(target_m[0] + "." + n)
    for name in block_names:
        print(name, flush=True)
        block = get_module(model, name)
        for n, m in block.named_modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.copy_(
                    quant_weight(m.weight, num_bits=num_bits, group_size=group_size, schema=schema))


def tokenize_function(examples):
    example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
    return example


@torch.no_grad()
def collate_batch(batch):
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


def get_module(model, key):
    """Get module from model by key name

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
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


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, num_bits, group_size, schema, grad=0):
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.orig_layer.weight.requires_grad_(True)
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


@torch.no_grad()
def unwrapper_block(block, num_bits, group_size, schema, grads):
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            orig_layer = m.orig_layer
            grad = 0
            if grads is not None:
                grad = grads[n]
            q_dq_weight = quant_weight(orig_layer.weight, num_bits, group_size, schema, grad)
            orig_layer.weight.data.copy_(q_dq_weight)
            orig_layer.weight.grad = None  ##clear grad
            set_module(block, n, orig_layer)


def collect_grad_and_zero(block):
    grads = {}
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            grad = -m.orig_layer.weight.grad
            grads[n] = copy.deepcopy(grad)
            m.orig_layer.weight.grad.zero_()
    return grads


def get_lr(step, total_steps, lr=0.01, warmup_step=0, lr_type="linear"):
    if warmup_step > 0 and step < warmup_step:
        return (lr - 0.01 * lr) * float(step) / warmup_step + 0.01 * lr

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


def sampling_inputs(input_ids, input_others, indices, model_name):
    if len(input_ids.shape) == 3:
        current_input_ids = input_ids[indices, :, :]
    else:
        n_samples = input_ids.shape[0] // seqlen
        current_input_ids = input_ids.view(n_samples, seqlen, -1)
        current_input_ids = current_input_ids[indices, :, :]
        current_input_ids = current_input_ids.reshape(-1, input.shape[-1])

    current_input_others = {}
    if "position_ids" in input_others.keys():
        current_input_others["position_ids"] = input_others["position_ids"]
    if "attention_mask" in input_others.keys():
        current_input_others["attention_mask"] = input_others["attention_mask"][indices, ...]
    if "bloom" in model_name:
        alibi = input_others["alibi"][indices, ...]
        current_input_others["alibi"] = alibi

    return current_input_ids, current_input_others


def block_forward(block, input_ids, input_others, model_name, amp=False, device=torch.device("cpu")):
    if input_ids.device != device:
        input_ids, input_others = move_to_device(input_ids, input_others, device)
    if "bloom" in model_name:
        attention_mask = input_others["attention_mask"]
        alibi = input_others["alibi"]
        alibi = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
        if amp and device != torch.device("cpu"):
            with autocast(device_type="cuda"):
                output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
        elif amp and device == torch.device("cpu"):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
        else:
            output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
    else:
        if amp and device != torch.device("cpu"):
            with autocast(device_type="cuda"):
                output = block.forward(input_ids, **input_others)
        elif amp and device == torch.device("cpu"):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block.forward(input_ids, **input_others)
        else:
            output = block.forward(input_ids, **input_others)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    return output


def move_to_device(input_ids, inputs_others=None, device=torch.device("cpu")):
    if input_ids != None:
        input_ids = input_ids.to(device)
    if inputs_others is not None:
        for key in inputs_others.keys():
            inputs_others[key] = inputs_others[key].to(device)
        return input_ids, inputs_others
    return input_ids


def quant_block(block, input_ids, input_others, num_bits, group_size, schema, q_input=None, args=None,
                device=torch.device("cpu")):
    best_loss = torch.finfo(torch.float).max
    mse_loss = torch.nn.MSELoss()
    grad = None
    if args.momentum > 0:
        grad_m = None
    output = []
    if not args.low_gpu_mem_usage and input_ids.device != device:
        input_ids, input_others = move_to_device(input_ids, input_others, device)

    with torch.no_grad():
        current_bs = args.train_bs
        for i in range(0, args.n_samples, current_bs):
            indices = torch.arange(i, i + current_bs).to(torch.long)
            tmp_input_ids, tmp_input_others = sampling_inputs(input_ids, input_others, indices, model_name)
            tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, model_name, args.amp, device)
            if args.low_gpu_mem_usage:
                tmp_output = tmp_output.to("cpu")
            output.append(tmp_output)

        output = torch.cat(output, dim=0)
    torch.cuda.empty_cache()
    if q_input is not None:
        input_ids = q_input
        if not args.low_gpu_mem_usage and input_ids.device != device:
            input_ids = input_ids.to(device)

    wrapper_block(block, num_bits, group_size, schema)
    search_iters = args.iters
    pick_samples = args.train_bs
    if len(input_ids.shape) == 3:
        n_samples = input_ids.shape[0]
    else:
        n_samples = input_ids.shape[0] // seqlen
    if args.sampler != "rand":
        indices = torch.randperm(n_samples)[:pick_samples]
    last_best_iter = 0
    for i in range(search_iters):
        if args.sampler == "rand":
            indices = torch.randperm(n_samples)[:pick_samples]

        total_loss = 0
        for _ in range(args.gradient_accumulate_steps):
            current_input_ids, current_input_others = sampling_inputs(input_ids, input_others, indices,
                                                                      model_name=args.model_name)
            if len(input_ids.shape) == 3:
                current_output = output[indices, :, :]
            else:
                current_output = output.view(n_samples, seqlen, -1)
                current_output = current_output[indices, :, :]
                current_output = current_output.reshape(-1, current_output.shape[-1])
            current_output = move_to_device(current_output, None, device)

            output_q = block_forward(block, current_input_ids, current_input_others, model_name, args.amp, device)
            if args.amp and device != torch.device("cpu"):
                with autocast(device_type="cuda"):
                    loss = mse_loss(output_q, current_output) * 1000
            elif args.amp:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    loss = mse_loss(output_q, current_output) * 1000
            else:
                loss = mse_loss(output_q, current_output)
            total_loss += loss.item()
            loss.backward()

        if total_loss < best_loss:
            best_loss = total_loss
            if not args.not_use_mse:
                # print(f"get better result at iter {i}, the loss is {total_loss}", flush=True)
                best_grad = copy.deepcopy(grad)
                last_best_iter = i
        if args.not_use_mse:
            best_grad = grad
        if not args.not_use_mse:
            if args.dynamic_max_gap > 0 and i - last_best_iter >= args.dynamic_max_gap:
                break
        new_grad = collect_grad_and_zero(block)

        warmup_step = int(args.iters * args.lr_wr)
        current_lr = get_lr(i, args.iters, args.lr, warmup_step, args.lr_decay_type)

        for key in new_grad.keys():
            new_grad[key] = torch.sign(new_grad[key]) * current_lr

        if grad is None:
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

    unwrapper_block(block, num_bits, group_size, schema, best_grad)
    if args.use_quant_input:
        with torch.no_grad():
            current_bs = args.train_bs
            start_index = 0
            q_outputs = []
            while 1:
                end_index = start_index + current_bs
                end_index = min(end_index, input_ids.shape[0])
                indices = torch.arange(start_index, end_index)
                current_input_ids, current_input_others = sampling_inputs(input_ids, input_others, indices,
                                                                          model_name=args.model_name)
                q_output = block_forward(block, current_input_ids, current_input_others, args.model_name,
                                         args.amp, device)
                q_outputs.append(q_output.to("cpu"))

                if end_index >= input_ids.shape[0]:
                    break
                else:
                    start_index = end_index
            q_outputs = torch.cat(q_outputs, dim=0)

        return q_outputs, output

    else:
        return None, output


class WrapperMultiblock(torch.nn.Module):
    def __init__(self, module_list):
        super(WrapperMultiblock, self).__init__()
        self.layers = torch.nn.ModuleList(module_list)

    def forward(self, x, **kwargs):
        hidden_states = x
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                **kwargs
            )
            hidden_states = layer_outputs
            if isinstance(hidden_states, tuple) or isinstance(hidden_states, list):
                hidden_states = layer_outputs[0]
        return hidden_states


def q_dq_weight_round(model: torch.nn.Module, inputs, block_names, num_bits=4, group_size=128, schema='asym',
                      n_blocks=1,
                      device=torch.device("cpu")):
    q_input = None
    torch.cuda.empty_cache()
    for n,m in model.named_parameters():
        m.requires_grad_(False)
    input_ids = inputs["input_ids"]
    inputs.pop('input_ids', None)
    input_others = inputs
    torch.cuda.empty_cache()
    for i in range(0, len(block_names), n_blocks):
        if n_blocks == 1:
            n = block_names[i]
            print(n, flush=True)
            m = get_module(model, n)
        else:
            names = block_names[i: i + n_blocks]
            print(names, flush=True)
            modules = [get_module(model, n) for n in names]
            m = WrapperMultiblock(modules)

        m = m.to(device)
        q_input, input_ids = quant_block(m, input_ids, input_others, num_bits=num_bits, group_size=group_size,
                                         schema=schema,
                                         q_input=q_input,
                                         args=args,
                                         device=device)
        m.to("cpu")
        torch.cuda.empty_cache()

    del q_input
    del input_ids
    del input_others
    del inputs

    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="/models/opt-125m"
    )

    parser.add_argument("--num_bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--train_bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=32, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default=0, type=str,
                        help="device gpu int number, or 'cpu' ")

    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

    parser.add_argument("--iters", default=400, type=int,
                        help=" iters")

    parser.add_argument("--dynamic_max_gap", default=0, type=int,
                        help="stop tuning if no best solution found within max_gap steps")

    parser.add_argument("--not_use_mse", action='store_true',
                        help=" whether use mse to get best grad")

    parser.add_argument("--use_quant_input", action='store_true',
                        help="whether to use the output of quantized block to tune the next block")

    parser.add_argument("--sampler", default="rand", type=str,
                        help="sampling type, rand or fix")

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

    parser.add_argument("--eval_fp16_baseline", action='store_true',
                        help="whether to eval FP16 baseline")

    parser.add_argument("--amp", action='store_true',
                        help="amp")

    parser.add_argument("--not_with_attention", action='store_true',
                        help="tuning with attention_mask input")

    parser.add_argument("--seqlen", default=512, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

    parser.add_argument("--n_blocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--n_samples", default=512, type=int,
                        help="number of samples")

    parser.add_argument("--lr_wr", default=0.0, type=float,
                        help="lr warmup ratio")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="low_gpu_mem_usage")

    parser.add_argument("--tasks", default=["lambada_openai"],
                        help="lm-eval tasks")

    # parser.add_argument("--tasks", default=["lambada_openai"],
    #                     help="lm-eval tasks")
    args = parser.parse_args()
    set_seed(args.seed)

    model_name = args.model_name
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)

    tasks = args.tasks

    if args.device == "cpu":
        device_str = "cpu"
    else:
        device_str = f"cuda:{int(args.device)}"
    cuda_device = torch.device(device_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True  ##low_cpu_mem_usage has impact to acc, changed the random seed?
    )
    model = model.eval()
    ##align wigh GPTQ to eval ppl
    if "opt" in model_name:
        seqlen = model.config.max_position_embeddings
        model.seqlen = model.config.max_position_embeddings
    else:
        seqlen = 2048
        model.seqlen = seqlen
    seqlen = args.seqlen

    if "llama" in model_name:
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.eval_fp16_baseline:
        model = model.to(cuda_device)
        eval_model(model, model_name, tokenizer, tasks=tasks, eval_bs=args.eval_bs)
        exit()

    if args.iters <= 0:
        q_dq_weight(model, num_bits=args.num_bits, group_size=args.group_size)
        model.half()
        model = model.to(cuda_device)
        eval_model(model, model_name, tokenizer, tasks=args.tasks, eval_bs=args.eval_bs)
        exit()

    dataset_name = "NeelNanda/pile-10k"
    if os.path.exists(dataset_name.split('/')[-1]):
        calib_dataset = load_from_disk(dataset_name.split('/')[-1])
    else:
        calib_dataset = load_dataset(dataset_name, split="train")
        calib_dataset.save_to_disk(dataset_name.split('/')[-1])

    calib_dataset = calib_dataset.shuffle(seed=args.seed)
    calib_dataset = calib_dataset.map(tokenize_function, batched=True)
    calib_dataset.set_format(type='torch', columns=['input_ids'])
    calib_dataloader = DataLoader(
        calib_dataset,
        batch_size=args.eval_bs,
        shuffle=False,
        collate_fn=collate_batch
    )
    target_m = None
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and 'ModuleList' in type(m).__name__:
            target_m = (n, m)

    block_names = []
    for n, m in target_m[1].named_children():
        block_names.append(target_m[0] + "." + n)
    seqlen = args.seqlen
    if args.amp and args.device != "cpu":
        model = model.half()
    elif args.amp and args.device == "cpu":
        model = model.to(torch.bfloat16)
    if not args.low_gpu_mem_usage:
        model = model.to(cuda_device)

    import time

    start_time = time.time()
    save_input_actor = SaveInputs(model, calib_dataloader, seqlen, block_names[0])
    inputs = save_input_actor.get_inputs(n_samples=args.n_samples)
    del save_input_actor
    if args.amp and args.device != "cpu":
        model = model.to("cpu").to(torch.float)

    model = model.to("cpu")
    torch.cuda.empty_cache()
    q_dq_weight_round(model, inputs, block_names, num_bits=args.num_bits, group_size=args.group_size,
                      n_blocks=args.n_blocks, device=cuda_device)
    end_time = time.time()
    print(end_time - start_time, flush=True)

    torch.cuda.empty_cache()
    model.eval()
    model.to(cuda_device)
    eval_model(model, model_name, tokenizer, tasks=args.tasks, eval_bs=args.eval_bs)
