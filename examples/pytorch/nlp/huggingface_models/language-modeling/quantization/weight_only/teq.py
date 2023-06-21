##CUDA_VISIBLE_DEVICES=3 python3 teq.py --model_name /models/opt-125m --training_step 1000 --lr 1e-3 --num_bits 4
import argparse

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
# import json


def quant_weight_asym(weight, num_bits=4):
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
    q = torch.clamp(torch.round(weight / scale) + zp, 0, maxq)
    return scale * (q - zp)


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


def quant_weight_actor(weight, num_bits, schema):
    assert num_bits > 0, "num_bits should be larger than 0"
    if schema == "sym":
        return quant_weight_sym(weight, num_bits)
    else:
        return quant_weight_asym(weight, num_bits)


def quant_weight(weight, num_bits=4, group_size=-1, schema="asym"):
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(weight, num_bits, schema=schema)

    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        weight = quant_weight_actor(weight, num_bits, schema=schema)
        weight = weight.reshape(orig_shape)
        return weight
    else:
        split_index = weight.shape[1] // group_size * group_size
        weight1 = weight[:, :split_index]
        weight1 = weight1.reshape(-1, group_size)
        weight1 = quant_weight_actor(weight1, num_bits, schema=schema)
        weight1 = weight1.reshape(orig_shape[0], split_index)
        weight2 = weight[:, split_index:]
        weight2 = quant_weight_actor(weight2, num_bits, schema=schema)
        weight = torch.cat([weight1, weight2], dim=1)
        return weight




class FakeAffineTensorQuantFunction(Function):
    """Fake version of affine quantization

    gemmlowp style scale+shift quantization. See more details in
    https://github.com/google/gemmlowp/blob/master/doc/quantization.md.

    We DO NOT recommend affine quantization on weights for performance reason. There might be value to affine quantize
    activation as it can be cancelled by bias and comes with no performance penalty. This functionality is only added
    for experimental purpose.
    """

    @staticmethod
    def forward(ctx, inputs, num_bits=4, group_size=1024, schema="asym"):
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
        # ctx.save_for_backward(inputs, min_range, max_range)
        return quant_weight(inputs, num_bits, group_size, schema)

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Args:
            ctx: Pytorch convention.
            grad_output: A tensor of gradient of outputs

        Returns:
            grad_inputs: A tensor of gradient
        """
        return grad_outputs, None, None, None
        # inputs, min_range, max_range = ctx.saved_tensors
        # min_range = min_range.unsqueeze(dim=-1)
        # max_range = max_range.unsqueeze(dim=-1)
        # zero = grad_outputs.new_zeros(1)
        # grad_inputs = torch.where((inputs <= max_range) * (inputs >= min_range), grad_outputs, zero)
        # return grad_inputs, None, None, None


class WrapperLinear(torch.nn.Module):
    def __init__(self, orig_layer, alpha, num_bits, group_size, schema):
        super(WrapperLinear, self).__init__()
        self.orig_layer = orig_layer
        self.alpha = alpha
        self.num_bits = num_bits
        self.group_size = group_size
        self.schema = schema
        self.tensor_quant = FakeAffineTensorQuantFunction().apply

    def forward(self, x):
        alpha = torch.clip(self.alpha, 1e-5)
        shape_len = len(x.shape) - 1
        shape = (1,) * shape_len + (-1,)
        alpha = alpha.to(x.device)
        x = x / alpha.view(shape)
        weight = self.orig_layer.weight
        alpha = alpha.to(weight.device)
        weight = weight * alpha.unsqueeze(dim=0)
        weight_q = FakeAffineTensorQuantFunction().apply(weight, self.num_bits, self.group_size, self.schema)
        return F.linear(x, weight_q, self.orig_layer.bias)


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


def absorb_scales(module, scale):  ##output channel
    """
    Absorb the scale to the layer at output channel
    :param layer_name: The module name
    :param scale: The scale to be absorbed
    :param alpha_key: The alpha passed to SQLinearWrapper
    :return:
    """
    layer = module
    # device = layer.device
    # dtype = layer.dtype

    if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.GroupNorm) or \
            isinstance(layer, torch.nn.InstanceNorm2d):
        if layer.affine:
            layer.weight *= scale
            layer.bias *= scale
        else:
            layer.affine = True
            weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
            layer.weight = torch.nn.Parameter(
                weight, requires_grad=False)
            bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False
                                            )
    elif isinstance(layer, torch.nn.LayerNorm):
        if layer.elementwise_affine:
            layer.weight *= scale
            layer.bias *= scale
        else:
            layer.elementwise_affine = True
            weight = torch.ones(layer.num_features, device=device, dtype=dtype) * scale
            layer.weight = torch.nn.Parameter(
                torch.ones(weight, requires_grad=False))
            bias = torch.zeros(layer.num_features, device=device, dtype=dtype)
            layer.bias = torch.nn.Parameter(
                bias, requires_grad=False)

    elif isinstance(layer, torch.nn.Conv2d):
        ##the order could not be changed
        if hasattr(layer, "bias") and (layer.bias != None):
            layer.bias *= scale
        scale = scale.view(scale.shape[0], 1, 1, 1)
        layer.weight *= scale

    elif isinstance(layer, torch.nn.Linear):
        if hasattr(layer, "bias") and (layer.bias != None):
            layer.bias *= scale
        scale = scale.view(scale.shape[0], 1)
        layer.weight *= scale


    elif layer.__class__.__name__ == "LlamaRMSNorm" \
            or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
        layer.weight *= scale

    else:
        print(f"found unsupported layer {type(layer)}, try to multiply scale to "
              f"weight and bias directly, this may introduce accuracy issue, please have a check ")
        if hasattr(layer, "weight") and layer.weight != None:
            layer.weight *= scale
        if hasattr(layer, "bias") and layer.bias != None:
            layer.bias *= scale


def scale_layer_weight(module, scale):  ##input channel
    """
    Scale the layer weights at input channel, depthwise conv output channel
    :param layer_name: The layer name
    :param scale: The scale to be multiplied
    :return:
    """

    scale = scale.view(1, scale.shape[0])
    module.weight = torch.nn.Parameter(module.weight * scale)
    return scale


def get_alpha_prefix(model_name, training_step, num_bits, group_size, betas, schema):
    prefix_name = f"trained_alphas_{model_name.split('/')[-1]}_{training_step}_groupsize_{group_size}_beta_{betas}_numb_{num_bits}_{schema}.pt"
    return prefix_name


def get_absorb_prefix(model_name):
    prefix_name = f"{model_name.split('/')[-1]}"
    could_absorb_layers_name = f"could_absorb_layers_{prefix_name}.pt"
    return could_absorb_layers_name


def get_absorb_layers(model_name):
    could_absorb_layers_name = get_absorb_prefix(model_name)
    if os.path.exists(could_absorb_layers_name):
        absorb_to_layer = torch.load(could_absorb_layers_name)
        return absorb_to_layer

    else:
        ##this model is for trace
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torchscript=True, low_cpu_mem_usage=True

        )
        from smooth_quant import GraphTrace
        tracer = GraphTrace()
        model.eval()
        model.to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        example_input = tokenizer("hello world", return_tensors='pt')
        absorb_to_layer, no_absorb_layers = tracer.get_absorb_to_layer(model, example_input['input_ids'], ['Linear'])
        torch.save(absorb_to_layer, could_absorb_layers_name)
        del model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        return absorb_to_layer


def wrapper_model(model, could_absorb_layers, quant_lm_head, num_bits, group_size, schema, device):
    trained_alphas = {}
    if not quant_lm_head:
        excluded_name = "lm_head"
        excluded_key = None
        for key, item in could_absorb_layers.items():
            if len(item) == 1 and excluded_name in item[0]:
                excluded_key = key
                break
        if excluded_key != None:
            could_absorb_layers.pop(excluded_key)  ##remove last layer

    for layer_norm in could_absorb_layers.keys():
        layer_0_name = could_absorb_layers[layer_norm][0]
        module = get_module(model, layer_0_name)
        if args.sqrt_w_init:
            weights = []
            for layer_name in could_absorb_layers[layer_norm]:
                module = get_module(model, layer_name)
                weights.append(module.weight)
            weights = torch.cat(weights, dim=0)
            max_value = torch.sqrt(torch.max(torch.abs(weights), dim=0).values)
            max_value[max_value == 0] = 1.0
            max_value = 1.0 / max_value
            alpha = torch.nn.Parameter(max_value)
            alpha = alpha.to(device)
        else:
            alpha = torch.nn.Parameter(torch.ones(module.weight.shape[1], device=device))

        trained_alphas[layer_norm] = alpha
        for layer_name in could_absorb_layers[layer_norm]:
            module = get_module(model, layer_name)
            wrapper_module = WrapperLinear(orig_layer=module, alpha=alpha, num_bits=num_bits, group_size=group_size,
                                           schema=schema)
            set_module(model, layer_name, wrapper_module)
    return trained_alphas


@torch.no_grad()
def transform(model, absorb_to_layers: dict, trained_alphas: dict):
    for ln_name, layer_names in absorb_to_layers.items():
        module = get_module(model, ln_name)
        scale = trained_alphas[ln_name]
        scale = torch.clip(scale, 1e-5)
        input_scale = 1.0 / scale
        if hasattr(module, "orig_layer"):
            module = module.orig_layer
        absorb_scales(module, input_scale)
        weight_scale = scale
        for layer_name in layer_names:
            layer_module = get_module(model, layer_name).orig_layer
            scale_layer_weight(layer_module, weight_scale)

        for layer_name in layer_names:
            layer_module = get_module(model, layer_name).orig_layer
            # quant_weight_tmp = quant_weight(layer_module.weight)
            # layer_module.weight.data.copy_(quant_weight_tmp)
            set_module(model, layer_name, layer_module)

    for n, m in model.named_modules():
        if isinstance(m, WrapperLinear):
            set_module(model, n, m.orig_layer)


@torch.no_grad()
def q_dq_weight(model: torch.nn.Module, num_bits, group_size, schema, quant_lm_head):
    for n, m in model.named_modules():
        if quant_lm_head:
            if isinstance(m, torch.nn.Linear):
                m.weight.data.copy_(
                    quant_weight(m.weight, num_bits=num_bits, group_size=group_size, schema=schema))
        else:
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
                m.weight.data.copy_(
                    quant_weight(m.weight, num_bits=num_bits, group_size=group_size, schema=schema))


if __name__ == '__main__':
    set_seed(42)
    # os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/models/huggingface"
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    parser.add_argument(
        "--model_name", nargs="?", default="facebook/opt-125m"
    )

    parser.add_argument("--group_size", default=-1, type=int,
                        help="weight_quantization config")

    parser.add_argument("--training_step", default=1000, type=int,
                        help="training step")
    parser.add_argument("--num_bits", default=4, type=int,
                        help="number of  bits")
    parser.add_argument("--gas", default=1, type=int,
                        help="gradient accumulate step")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="eval batch_size")
    parser.add_argument("--device", default=0, type=str,
                        help="device gpu int number, or 'cpu' ")

    parser.add_argument("--betas", default=0.9, type=float,
                        help="adam betas ")

    parser.add_argument("--lr", default=1e-3, type=float,
                        help="learning rate")

    parser.add_argument("--fp16", action='store_true',
                        help=" fp16 ")
    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

    parser.add_argument("--quant_lm_head", action='store_true',
                        help=" quant lm head")

    parser.add_argument("--sqrt_w_init", action='store_true',
                        help="sqrt_w_init")

    # parser.add_argument("--layer_wise_training", action='store_true',
    #                     help="layer_wise_training")

    args = parser.parse_args()
    schema = "asym"
    if args.sym:
        schema = "sym"
    if args.device == "cpu":
        device_str = "cpu"
    else:
        device_str = f"cuda:{int(args.device)}"
    device = torch.device(device_str)
    model_name = args.model_name
    if args.training_step > 0:  ##wrapper module
        absorb_to_layers = get_absorb_layers(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True,
    )

    if args.training_step > 0:

        model = model.to(device)
        trained_alphas = wrapper_model(model, absorb_to_layers, args.quant_lm_head, args.num_bits, args.group_size,
                                       schema, model.device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        seqlen = 512


        def tokenize_function(examples):
            example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
            return example


        dataset_name = "NeelNanda/pile-10k"
        if os.path.exists(dataset_name.split('/')[-1]):
            calib_dataset = load_from_disk(dataset_name.split('/')[-1])
        else:
            calib_dataset = load_dataset(dataset_name, split="train")
            calib_dataset.save_to_disk(dataset_name.split('/')[-1])

        # calib_dataset = load_dataset(
        #     'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        # )
        calib_dataset = calib_dataset.shuffle(seed=42)
        calib_dataset = calib_dataset.map(tokenize_function, batched=True)
        calib_dataset.set_format(type='torch', columns=['input_ids'])

        calib_dataloader = DataLoader(
            calib_dataset,
            batch_size=1,
            shuffle=False,
            ##collate_fn=collate_batch,
        )



        trained_alphas_list = []
        for item in trained_alphas.items():
            trained_alphas_list.append(item[1])
        # parameters = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = torch.optim.Adam(trained_alphas_list, lr=args.lr, weight_decay=0, betas=[0.9, args.betas])
        # optimizer = optim.SGD(), lr=0.1

        from transformers import get_scheduler

        train_step = args.training_step
        gradient_accumulation_steps = args.gas
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(train_step * 0.05) // gradient_accumulation_steps,
            num_training_steps=train_step // gradient_accumulation_steps,
        )

        cnt = 1
        import time

        results = {}
        start_time = time.time()
        model.train()
        print("start training", flush=True)
        while 1:
            for inputs in calib_dataloader:
                if cnt % 100 == 0:
                    end_time = time.time() - start_time
                    print(cnt, end_time, flush=True)
                if isinstance(inputs, dict):
                    input_id = inputs["input_ids"]
                else:
                    input_id = inputs[0]
                ##print(input_id.shape)


                input_id = input_id.to(device)
                output = model(input_id, labels=input_id)
                loss = output[0] / gradient_accumulation_steps
                loss.requires_grad_(True)

                # if args.layer_wise_training:
                #     loss.to("cpu")
                    # model.to("cpu")
                loss.backward()
                # print(output[0])
                ##print(f"{cnt}/{len(calib_dataloader)}, {output.loss}")
                if cnt % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                if cnt == args.training_step:
                    break
                cnt += 1
            if cnt == args.training_step:
                break
        print("finish training", flush=True)
        for key, item in trained_alphas.items():
            item.to("cpu")
        alpha_prefix = get_alpha_prefix(model_name, args.training_step, args.num_bits, args.group_size, args.betas,
                                        schema)
        torch.save(trained_alphas, alpha_prefix)
        model.eval()
        transform(model, absorb_to_layers=absorb_to_layers, trained_alphas=trained_alphas)
    model.eval()
    model.to(device)
    q_dq_weight(model, num_bits=args.num_bits, group_size=args.group_size, schema=schema,
                quant_lm_head=args.quant_lm_head)
    tmp_dtype = "float32"
    if args.fp16:
        model = model.half()
        tmp_dtype = "float16"
    results = lm_evaluate(model="hf-causal",
                          model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype={tmp_dtype}',
                          user_model=model, tasks=["lambada_openai"],
                          device=str(model.device),
                          batch_size=args.eval_batch_size)
