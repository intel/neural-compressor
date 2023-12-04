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
from collections import UserDict
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["HF_HOME"] = "/models/huggingface"


# os.environ['TRANSFORMERS_OFFLINE'] = '0'


class FakeAffineTensorQuantFunction(Function):
    @staticmethod
    def forward(ctx, inputs, num_bits=4, group_size=128, schema="asym", grad=0):
        return quant_weight(inputs, num_bits, group_size, schema, grad)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None, None, None


def round_ste(x: torch.Tensor):
    """
    cp from https://github.com/OpenGVLab/OmniQuant/blob/main/quantize/quantizer.py
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def quant_weight_asym(weight, num_bits=4, grad=0, min_scale=0, max_scale=0):
    maxq = torch.tensor(2 ** num_bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device)
    if isinstance(min_scale, torch.Tensor):
        wmin_tmp = torch.minimum(weight.min(1)[0], zeros)
        wmax_tmp = torch.maximum(weight.max(1)[0], zeros)
        wmin_tmp *= (min_scale + 1.0)
        wmax_tmp *= (max_scale + 1.0)
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        wmin = torch.minimum(weight.min(1)[0], zeros)
        wmax = torch.maximum(weight.max(1)[0], zeros)
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = (wmax - wmin) / maxq
    zp = round_ste(-wmin / scale)
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + grad)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp)


def quant_weight_sym(weight, num_bits=4, grad=0, min_scale=0, max_scale=0):
    maxq = torch.tensor(2 ** (num_bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-2 ** (num_bits - 1)).to(weight.device)
    if num_bits == 1:
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)

    wmax = torch.abs(weight).max(1)[0]
    wmax *= (1 + max_scale)
    tmp = (wmax == 0)
    wmax[tmp] = +1
    scale = wmax / ((maxq - minq) / 2)
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(round_ste(weight / scale + grad), minq, maxq)
    return scale * q


def quant_weight_actor(weight, num_bits, schema, grad, min_scale, max_scale):
    assert num_bits > 0, "num_bits should be larger than 0"
    if schema == "sym":
        return quant_weight_sym(weight, num_bits, grad, min_scale, max_scale)
    else:
        return quant_weight_asym(weight, num_bits, grad, min_scale, max_scale)


def quant_weight(weight, num_bits=4, group_size=-1, schema="asym", grad=0, min_scale=0, max_scale=0):
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(weight, num_bits, schema=schema, grad=grad, min_scale=min_scale, max_scale=max_scale)
    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if isinstance(grad, torch.Tensor):
            grad = grad.reshape(-1, group_size)

        weight = quant_weight_actor(weight, num_bits, schema=schema, grad=grad, min_scale=min_scale,
                                    max_scale=max_scale)

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
        if isinstance(min_scale, torch.Tensor):
            min_scale_1 = min_scale[:, :weight.shape[1] // group_size]
            min_scale_2 = min_scale[:, weight.shape[1] // group_size:]
            max_scale_1 = max_scale[:, :weight.shape[1] // group_size]
            max_scale_2 = max_scale[:, weight.shape[1] // group_size:]
        else:
            min_scale_1 = min_scale
            min_scale_2 = min_scale
            max_scale_1 = max_scale
            max_scale_2 = max_scale

        weight1 = quant_weight_actor(weight1, num_bits, schema=schema, grad=grad1, min_scale=min_scale_1,
                                     max_scale=max_scale_1)
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2 = quant_weight_actor(weight2, num_bits, schema=schema, grad=grad2, min_scale=min_scale_2,
                                     max_scale=max_scale_2)
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
        def forward(_, hidden_states, *positional_args, **kwargs):  ##This may have bug for other models
            dim = int((hasattr(self.model, "config") and "chatglm" in self.model.config.model_type))
            if name in self.inputs:
                data = torch.cat([self.inputs[name]["input_ids"], hidden_states.to("cpu")], dim=dim)
                self.inputs[name]["input_ids"] = data
            else:
                self.inputs[name] = {}
                self.inputs[name]["input_ids"] = hidden_states.to("cpu")

            if "positional_inputs" not in self.inputs[name]:
                self.inputs[name]["positional_inputs"] = []
            for idx, item in enumerate(positional_args):
                print(idx, item)
                self.inputs[name]["positional_inputs"] = move_input_to_device(positional_args)

            for key in kwargs.keys():
                if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) or (key == "alibi"):
                    if "attention_mask" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if kwargs[key] is not None:
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key] = torch.cat(
                                    [self.inputs[name][key], kwargs[key].to("cpu")], dim=0
                                )
                            else:
                                self.inputs[name][key] = kwargs[key].to("cpu")
                    elif "alibi" in key:
                        if key not in self.inputs[name].keys():
                            self.inputs[name][key] = None
                        if isinstance(kwargs[key], torch.Tensor):
                            alibi = kwargs[key]
                            batch = kwargs["attention_mask"].shape[0]
                            alibi = alibi.reshape(batch, -1, alibi.shape[1], alibi.shape[2])
                            if self.inputs[name][key] is not None:
                                self.inputs[name][key] = torch.cat([self.inputs[name][key], alibi.to("cpu")], dim=0)
                            else:
                                self.inputs[name][key] = alibi.to("cpu")
                    elif key not in self.inputs[name].keys():
                        self.inputs[name][key] = move_input_to_device(kwargs[key], device=torch.device("cpu"))
            raise NotImplementedError

        return forward

    @torch.no_grad()
    def get_inputs(self, n_samples=512):
        total_cnt = 0
        self._replace_forward()
        for data in self.dataloader:
            if data is None:
                continue

            input_ids = data["input_ids"].to(self.model.device)
            if input_ids.shape[-1] < self.seqlen:
                continue
            if total_cnt + input_ids.shape[0] > n_samples:
                input_ids = input_ids[: n_samples - total_cnt, ...]
            try:
                self.model(input_ids)
            except NotImplementedError:
                pass
            except Exception as error:
                logger.error(error)
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
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    module = model
    name_list = key.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
            module = module
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)


def get_scale_shape(weight, group_size):
    if group_size == -1 or weight.shape[1] < group_size:
        shape = (weight.shape[0])
    else:
        shape = (weight.shape[0] * ((weight.shape[1] + group_size - 1) // group_size))

    return shape


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, num_bits, group_size, schema, enable_minmax_tuning=True):
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.schema = schema
        self.value = torch.nn.Parameter(torch.zeros(self.orig_layer.weight.shape, device=self.orig_layer.weight.device),
                                        requires_grad=True)
        self.enable_minmax_tuning = enable_minmax_tuning
        shape = get_scale_shape(self.orig_layer.weight, group_size)

        if self.enable_minmax_tuning:
            self.min_scale = torch.nn.Parameter(torch.zeros(shape, device=self.orig_layer.weight.device),
                                                requires_grad=True)
            self.max_scale = torch.nn.Parameter(torch.zeros(shape, device=self.orig_layer.weight.device),
                                                requires_grad=True)
        else:
            self.min_scale = torch.tensor(0, device=self.orig_layer.weight.device)
            self.max_scale = torch.tensor(0, device=self.orig_layer.weight.device)

    @torch.no_grad()
    def update_grad(self, grad):
        self.value.data.copy_(grad)

    @torch.no_grad()
    def update_minmax_grad(self, min_scale, max_scale):
        self.min_scale.copy_(min_scale)
        self.max_scale.copy_(max_scale)

    def forward(self, x):
        weight = self.orig_layer.weight
        self.min_scale.data.copy_(torch.clamp(self.min_scale.data, -1, 0))
        self.max_scale.data.copy_(torch.clamp(self.max_scale.data, -1, 0))
        weight_q = quant_weight(weight, self.num_bits, self.group_size, self.schema,
                                self.value, self.min_scale, self.max_scale)
        # if self.enable_minmax_tuning:
        #     weight_q = quant_weight(weight, self.num_bits, self.group_size, self.schema,
        #                             self.value)
        #
        # else:
        #     weight_q = FakeAffineTensorQuantFunction().apply(weight, self.num_bits, self.group_size, self.schema,
        #                                                      self.value)  ##mainly for reproducing the data of paper
        return F.linear(x, weight_q, self.orig_layer.bias)


def wrapper_block(block, num_bits, group_size, schema, enable_minmax_tuning):
    for n, m in block.named_modules():
        if isinstance(m, torch.nn.Linear):
            new_m = WrapperLinear(m, num_bits, group_size, schema, enable_minmax_tuning=enable_minmax_tuning)
            set_module(block, n, new_m)


@torch.no_grad()
def unwrapper_block(block, num_bits, group_size, schema, grads, min_scale_grads, max_scale_grads):
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            orig_layer = m.orig_layer
            grad = 0
            min_scale_grad = 0
            max_scale_grad = 0
            if isinstance(grads, dict):
                grad = grads[n]
            if isinstance(min_scale_grads, dict):
                min_scale_grad = min_scale_grads[n]
                min_scale_grad = torch.clamp(min_scale_grad, -1, 0)
            if isinstance(max_scale_grads, dict):
                max_scale_grad = max_scale_grads[n]
                max_scale_grad = torch.clamp(max_scale_grad, -1, 0)

            q_dq_weight = quant_weight(orig_layer.weight, num_bits, group_size, schema, grad, min_scale_grad,
                                       max_scale_grad)
            orig_layer.weight.data.copy_(q_dq_weight)
            orig_layer.weight.grad = None  ##clear grad
            set_module(block, n, orig_layer)


def collect_grad_and_zero(block):
    grads = {}
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            grad = -m.orig_layer.value.grad
            grads[n] = copy.deepcopy(grad)
            m.orig_layer.weight.grad.zero_()
    return grads


def get_lr(step, total_steps, lr=0.01, warmup_step=0, lr_type="linear"):
    if warmup_step > 0 and step < warmup_step:
        return (lr - 0.01 * lr) * float(step) / warmup_step + 0.01 * lr

    if lr_type == "const":
        current_lr = lr
    elif lr_type == "linear":
        current_lr = lr - float(step - warmup_step) / (total_steps - warmup_step) * lr
    elif lr_type == "cos":
        import math
        current_lr = math.cos(float(step - warmup_step) / (total_steps - warmup_step) * math.pi / 2) * lr
    else:
        raise NotImplemented
    return current_lr



def sampling_inputs(input_ids, input_others, indices, seqlen):
    if len(input_ids.shape) == 3:
        if int(len(input_others["positional_inputs"]) > 0):
            current_input_ids = input_ids[:, indices, :]
        else:
            current_input_ids = input_ids[indices, :, :]
    else:
        n_samples = input_ids.shape[0] // seqlen
        current_input_ids = input_ids.view(n_samples, seqlen, -1)
        current_input_ids = current_input_ids[indices, :, :]
        current_input_ids = current_input_ids.reshape(-1, input.shape[-1])

    current_input_others = {}
    current_input_others["positional_inputs"] = input_others["positional_inputs"]
    for key in input_others.keys():
        if "attention_mask" in key or "alibi" in key:
            current_input_others[key] = None
            if input_others[key] is not None:
                current_input_others[key] = input_others[key][indices, ...]
        else:
            current_input_others[key] = input_others[key]

    return current_input_ids, current_input_others


def move_input_to_device(input, device=torch.device("cpu")):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = move_input_to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res = []
        for inp in input:
            input_res.append(move_input_to_device(inp, device))
        input = input_res
    return input


def block_forward(block, input_ids, input_others, amp=False, device=torch.device("cpu")):
    if input_ids.device != device:
        # input_ids, input_others = move_to_device(input_ids, input_others, device)
        input_ids = move_input_to_device(input_ids, device)
        input_others = move_input_to_device(input_others, device)
    if "alibi" in input_others.keys():
        attention_mask = input_others["attention_mask"]
        alibi = input_others["alibi"]
        alibi = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
        if amp and device != torch.device("cpu"):
            with autocast(device_type="cuda"):
                output = block(
                    input_ids, attention_mask=attention_mask, alibi=alibi
                )  ##TODO is this correct for all models with alibi?
        elif amp and device == torch.device("cpu"):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
        else:
            output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
    else:
        input_tuple = input_others.pop("positional_inputs", None)
        if amp and device != torch.device("cpu"):
            with autocast(device_type="cuda"):
                output = block.forward(input_ids, *input_tuple, **input_others)
        elif amp and device == torch.device("cpu"):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block.forward(input_ids, *input_tuple, **input_others)
        else:
            output = block.forward(input_ids, *input_tuple, **input_others)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    return output





def collect_round_grad(block):
    grads = {}
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            grad = m.value.data
            grads[n] = copy.deepcopy(grad)
    return grads


def collect_minmax_grad(block):
    min_grads = {}
    max_grads = {}
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            min_grads[n] = copy.deepcopy(torch.clamp(m.min_scale.data, -1, 0))
            max_grads[n] = copy.deepcopy(torch.clamp(m.max_scale.data, -1, 0))
    return min_grads, max_grads

@torch.no_grad()
def get_block_outputs(block, input_ids, input_others, bs, device, cache_device, batch_dim,args):
    output = []
    for i in range(0, args.n_samples, bs):
        indices = torch.arange(i, i + bs).to(torch.long)
        tmp_input_ids, tmp_input_others = sampling_inputs(input_ids, input_others, indices, args.seqlen)
        tmp_output = block_forward(block, tmp_input_ids, tmp_input_others, args.amp, device).to(cache_device)
        output.append(tmp_output)
    output = torch.cat(output, dim=batch_dim)
    torch.cuda.empty_cache()
    return output


@torch.no_grad()
def get_batch_dim(input_others):
    dim = int(len(input_others["positional_inputs"]) > 0)
    return dim


def quant_block(block, input_ids, input_others, num_bits, group_size, schema, q_input=None, args=None,
                device=torch.device("cpu")):
    best_loss = torch.finfo(torch.float).max
    mse_loss = torch.nn.MSELoss()
    grad = None
    grad_m = None
    min_scale_grad = None
    max_scale_grad = None
    output = []
    batch_dim = get_batch_dim(input_others)
    if not args.low_gpu_mem_usage and input_ids.device != device:
            # input_ids, input_others = move_to_device(input_ids, input_others, device)
            input_ids = move_input_to_device(input_ids, device)  ##move input data to GPU
            input_others = move_input_to_device(input_others, device)
    cache_device = device
    if args.low_gpu_mem_usage:
        cache_device = "cpu"
    output = get_block_outputs(block, input_ids, input_others, args.train_bs, device, cache_device, batch_dim,args)
    if q_input is not None:
        input_ids = q_input
        if not args.low_gpu_mem_usage and input_ids.device != device:
            input_ids = input_ids.to(device)

    wrapper_block(block, num_bits, group_size, schema, args.enable_minmax_tuning)

    round_params = []
    minmax_params = []
    for n, m in block.named_modules():
        if isinstance(m, WrapperLinear):
            round_params.append(m.value)
            minmax_params.append(m.min_scale)
            minmax_params.append(m.max_scale)
    from sign_sgd import SGD
    # optimizer = SGD(params, lr=0.0025)
    if args.enable_minmax_tuning:
        optimizer = SGD([{'params': round_params}, {'params': minmax_params, 'lr': args.min_max_lr}],
                        lr=args.lr, weight_decay=0)
    else:
        optimizer = SGD(round_params,
                        lr=args.lr, weight_decay=0)

    lr_schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=args.iters,
                                                    verbose=False)

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
                                                                      seqlen=args.seqlen)
            if len(input_ids.shape) == 3:
                if batch_dim == 0:
                    current_output = output[indices, :, :]
                elif batch_dim == 1:
                    current_output = output[:, indices, :]
                else:
                    current_output = output[:, :, indices]
            else:
                current_output = output.view(n_samples, seqlen, -1)
                current_output = current_output[indices, :, :]
                current_output = current_output.reshape(-1, current_output.shape[-1])
            current_output = move_input_to_device(current_output, device)

            output_q = block_forward(block, current_input_ids, current_input_others, args.amp, device)
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
                best_grad = collect_round_grad(block)
                best_min_scale_grad, best_max_scale_grad = collect_minmax_grad(block)
                last_best_iter = i
        if args.not_use_mse:
            best_grad = grad
            best_min_scale_grad = min_scale_grad
            best_max_scale_grad = max_scale_grad

        if not args.not_use_mse:
            if args.dynamic_max_gap > 0 and i - last_best_iter >= args.dynamic_max_gap:
                break
        optimizer.step()##TODO  scale grad for other optimizer
        optimizer.zero_grad()
        lr_schedule.step()
    unwrapper_block(block, num_bits, group_size, schema, best_grad, best_min_scale_grad, best_max_scale_grad)
    if args.use_quant_input:
        q_outputs = get_block_outputs(
            block, input_ids, input_others, args.train_bs, device,cache_device, batch_dim,args
        )

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
    for n, m in model.named_parameters():
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

    parser.add_argument("--lr", default=0.05, type=float,
                        help="step size")

    parser.add_argument("--min_max_lr", default=0.05, type=float,
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

    parser.add_argument("--enable_minmax_tuning", action='store_true',
                        help="enable_tuning_minmax")

    # parser.add_argument("--tasks", default=["lambada_openai", "hellaswag", "winogrande", "piqa"],
    #                     help="lm-eval tasks")

    # parser.add_argument("--tasks", default=["lambada_openai"],
    #                     help="lm-eval tasks")
    #
    # parser.add_argument("--tasks",
    #                     default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
    #                              'coqa', 'truthfulqa_mc', 'openbookqa', 'boolq', 'rte', 'arc_easy', 'arc_challenge',
    #                              'hendrycksTest-*', 'wikitext', 'drop', 'gsm8k'],##all
    #                     help="lm-eval tasks")  # "truthfulqa_gen"

    # parser.add_argument("--tasks",
    #                     default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
    #                              'coqa', 'truthfulqa_mc', 'openbookqa', 'boolq', 'rte', 'arc_easy', 'arc_challenge',
    #                              'hendrycksTest-*', 'wikitext', 'drop', 'gsm8k'],##all
    # parser.add_argument("--tasks",
    #                     default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
    #                              "hendrycksTest-*", "wikitext", "truthfulqa_mc", "openbookqa", "boolq", "rte",
    #                              "arc_easy", "arc_challenge"],
    #                     help="lm-eval tasks")  # "truthfulqa_gen"

    parser.add_argument("--tasks", default=["lambada_openai"],
                        help="lm-eval tasks")

    parser.add_argument("--output_dir", default="./tmp_optround", type=str,
                        help="Where to store the final model.")

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
        model_name, low_cpu_mem_usage=True,trust_remote_code=True,  ##low_cpu_mem_usage has impact to acc, changed the random seed?
    )
    model = model.eval()
    # align wigh GPTQ to eval ppl
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    excel_name = f"{model_name}_{args.num_bits}_{args.group_size}"
    if args.eval_fp16_baseline:
        if not args.low_gpu_mem_usage:
            model = model.to(cuda_device)
        excel_name += "_fp16.xlsx"
        eval_model(output_dir=model_name, model=model, tokenizer=tokenizer, tasks=args.tasks, \
                   eval_bs=args.eval_bs, use_accelerate=args.low_gpu_mem_usage, device=cuda_device,
                   eval_orig_float=True, excel_file=excel_name)
        exit()

    if args.iters <= 0:
        print("eval rtn", flush=True)
        excel_name += "_optround.xlsx"
        q_dq_weight(model, num_bits=args.num_bits, group_size=args.group_size)
        model.half()
        if not args.low_gpu_mem_usage:
            model = model.to(cuda_device)
        eval_model(output_dir=args.output_dir, model=model, tokenizer=tokenizer, tasks=args.tasks, \
                   eval_bs=args.eval_bs, use_accelerate=args.low_gpu_mem_usage, device=cuda_device,
                   excel_file=excel_name)
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
    model.half()
    model.eval()
    output_dir = args.output_dir + "_" + args.model_name.split('/')[-1] + f"_w{args.num_bits}_g{args.group_size}"

    # model.to(cuda_device)
    # eval_model(model, model_name, tokenizer, tasks=args.tasks, eval_bs=args.eval_bs)
    excel_name = f"{output_dir}_result.xlsx"
    output_dir += "/"
    print(excel_name, flush=True)
    eval_model(output_dir=output_dir, model=model, tokenizer=tokenizer, tasks=args.tasks, \
               eval_bs=args.eval_bs, use_accelerate=args.low_gpu_mem_usage, device=cuda_device, excel_file=excel_name,
               limit=None)
