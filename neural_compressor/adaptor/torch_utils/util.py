#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Util Class and Functions."""
import copy
import re
import numpy as np
from collections import UserDict
from packaging.version import Version
from ...utils import logger
from ...utils.utility import LazyImport, CpuInfo

tqdm = LazyImport("tqdm")
torch = LazyImport("torch")

def get_embedding_contiguous(model):
    """This is a helper function for nn.Embedding, and it will get input contiguous.

    Args:
        model (object): the input model

    Returns:
        None
    """
    def contiguous_hook(module, input):
        embeddings = input[0].contiguous()
        modified_input = (embeddings, *input[1:])
        return modified_input

    for child in model.modules():
        child_type = child.__class__.__name__
        if child_type == 'Embedding':
            child.register_forward_pre_hook(contiguous_hook)


def is_fused_module(module):
    """This is a helper function for `_propagate_qconfig_helper` to detect if this module is fused.

    Args:
        module (object): the input module

    Returns:
        (bool): is fused or not
    """
    op_type = str(type(module))
    if 'fused' in op_type:
        return True
    else:
        return False


def _set_input_scale_hook(model, op_cfgs):
    """Insert hooks to observer input scale and zeropoint.

    Args:
        model (object): the input model
        op_cfgs (dict): dictionary of quantization configure for each op

    Returns:
        hook_list (list): the input observer hooks
    """
    def input_scale_hook(module, input):
        module.input_observer = module.qconfig.activation()
        module.input_observer(input[0])
        return input

    def output_scale_hook(module, input, output):
        module.output_observer = module.qconfig.activation()
        module.output_observer(output)
        return output

    def ConvReLU2d_scale_hook(module, input):
        module.input_observer = module.qconfig.activation()
        module.input_observer(input[0])
        output = module._conv_forward(input[0], module.weight_fake_quant(module.weight), module.bias)
        module.output_observer = module.qconfig.activation()
        module.output_observer(output)
        return input

    def LinearReLU_scale_hook(module, input):
        import torch.nn.functional as F
        module.input_observer = module.qconfig.activation()
        module.input_observer(input[0])
        output = F.linear(input[0], module.weight_fake_quant(module.weight), module.bias)
        module.output_observer = module.qconfig.activation()
        module.output_observer(output)
        return input

    hook_list = []
    for name, module in model.named_modules():
        if 'Conv' in str(module.__class__.__name__) or \
          'Linear' in str(module.__class__.__name__):
            if not hasattr(module, 'qconfig') or not module.qconfig:
                continue
            from torch.nn.intrinsic.qat import ConvBn2d, ConvReLU2d, ConvBnReLU2d, LinearReLU
            if type(module) in [ConvBn2d, ConvBnReLU2d]:
                handle_in = module.register_forward_pre_hook(input_scale_hook)
                # module[0] == torch.nn.BatchNorm2d
                module[0].qconfig = module.qconfig
                handle_out = module[0].register_forward_hook(output_scale_hook)
                hook_list.extend([handle_in, handle_out])
            elif type(module) in [ConvReLU2d]:
                handle_in_out = module.register_forward_pre_hook(ConvReLU2d_scale_hook)
                hook_list.extend([handle_in_out])
            elif type(module) in [LinearReLU]:
                handle_in_out = module.register_forward_pre_hook(LinearReLU_scale_hook)
                hook_list.extend([handle_in_out])
            else:
                if is_fused_module(module):
                    continue
                handle_in = module.register_forward_pre_hook(input_scale_hook)
                handle_out = module.register_forward_hook(output_scale_hook)
                hook_list.extend([handle_in, handle_out])
    return hook_list


def _get_input_scale(model, hook_list):
    """Fetch input scale and zeropoint from observer.

    Args:
        model (object): the input model
        hook_list (list): the input observer hooks

    Returns:
        input_scale_info (dict): the input scale and zero_point of each modules
    """
    scale_info = {}
    for name, module in model.named_modules():
        from torch.nn.intrinsic.qat import ConvBn2d, ConvBnReLU2d
        if type(module) in [ConvBn2d, ConvBnReLU2d]:
            if hasattr(module, "input_observer") and hasattr(module[0], "output_observer"):
                scale_in, zero_point_in = module.input_observer.calculate_qparams()
                scale_out, zero_point_out = module[0].output_observer.calculate_qparams()
                scale_info[name] = {
                    'input_scale': float(scale_in),
                    'input_zeropoint': int(zero_point_in),
                    'output_scale': float(scale_out),
                    'output_zeropoint': int(zero_point_out)
                }
                del module.input_observer, module[0].output_observer
        elif hasattr(module, "input_observer") and hasattr(module, "output_observer"):
            scale_in, zero_point_in = module.input_observer.calculate_qparams()
            scale_out, zero_point_out = module.output_observer.calculate_qparams()
            scale_info[name] = {
                'input_scale': float(scale_in),
                'input_zeropoint': int(zero_point_in),
                'output_scale': float(scale_out),
                'output_zeropoint': int(zero_point_out)
            }
            del module.input_observer, module.output_observer
    for h in hook_list:
        h.remove()
    return scale_info


def collate_torch_preds(results):
    """Fetch collated results.

    Args:
        result (list): input result

    Returns:
        collate_results (list): collated results
    """
    batch = results[0]
    if isinstance(batch, list):
        results = zip(*results)
        collate_results = []
        for output in results:
            output = [
                batch.numpy() if isinstance(batch, torch.Tensor) else batch
                for batch in output
            ]
            collate_results.append(np.concatenate(output))
    elif isinstance(batch, torch.Tensor):
        results = [
            batch.numpy() if isinstance(batch, torch.Tensor) else batch
            for batch in results
        ]
        collate_results = np.concatenate(results)
    return collate_results


def input2tuple(input):
    """This is a helper function to converting a inputting dict values or a list to a tuple.

    Args:
        input (list or dict).

    Returns:
        A tuple.
    """
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = tuple(input.values())
    elif isinstance(input, list) or isinstance(input, tuple):
        output = tuple(input)
    else:
        output = input
    return output


def append_attr(fx_model, model):
    """This is a helper method to append attributes for the symbolic traced model.

    Args:
        fx_model (torch.fx.GraphModule): The symbolic traced model.
        model (torch.nn.Module): The original model.

    Returns:
        fx_model (dir): The symbolic traced model with additional attributes.
    """
    fx_attr = dir(fx_model)
    org_attr = dir(model)
    ignore_match_patterns = [r"_", r"quant", r"dequant", r"weight", 
                            r"bias", r'activation_post_process']
    ignore_search_patterns = [r"_scale_", r"_zero_point_", 
                            r'_activation_post_process_']
    attr_names = []
    for i in org_attr:
        if i not in fx_attr and \
          not any([re.match(p, i) for p in ignore_match_patterns]) and \
          not any([re.search(p, i) for p in ignore_search_patterns]) :
            attr_names.append(i)
    for name in attr_names:
        attr = getattr(model, name)
        if isinstance(attr, torch.nn.Module) or \
          isinstance(attr, torch.quantization.qconfig.QConfig):
            continue
        setattr(fx_model, name, attr)
    return fx_model


def generate_activation_observer(scheme, algorithm): # pragma: no cover
    """This is a helper method to generate an activation observer.

    Args:
        scheme (str): Quantization scheme to be used.
        algorithm (str): What algorithm for computing the quantization parameters based on.

    Returns:
        An observer.
    """
    kl_activation_observer = {
                    'name': 'HistogramObserver', 
                    'bins': 2048,
                    'upsample_rate': 128,
                    'dtype': 'torch.quint8',
                    'qscheme': 'torch.per_tensor_affine',
                    'reduce_range': False,
                    'quant_min': 0,
                    'quant_max': 255
                    }
    minmax_activation_observer = {
                    "name": "MinMaxObserver",
                    "dtype": "torch.quint8",
                    "qscheme": "torch.per_tensor_affine",
                    "reduce_range": False,
                    "quant_min": 0,
                    "quant_max": 255
                }
    REDUCE_RANGE = False if CpuInfo().vnni else True
    if REDUCE_RANGE:
        minmax_activation_observer["reduce_range"] = REDUCE_RANGE
        kl_activation_observer["reduce_range"] = REDUCE_RANGE
    if scheme == "sym":
        minmax_activation_observer["qscheme"] = "torch.per_tensor_symmetric"
        minmax_activation_observer["dtype"] = "torch.qint8"
        minmax_activation_observer["quant_min"] = -128
        minmax_activation_observer["quant_max"] = 127
        kl_activation_observer["qscheme"] = "torch.per_tensor_symmetric"
        kl_activation_observer["dtype"] = "torch.qint8"
        kl_activation_observer["quant_min"] = -128
        kl_activation_observer["quant_max"] = 127
    if algorithm == "kl":
        return kl_activation_observer
    if algorithm == "minmax":
        return minmax_activation_observer

def check_cfg_and_qconfig(tune_cfg, cfgs, op_infos_from_cfgs, output_tensor_ids_op_name): # pragma: no cover
    """Check configs and quantization configs.

    Args:
        tune_cfg (dict): dictionary of quantization configuration.
        cfgs (dict): the input configs.
        op_infos_from_cfgs (dict): op infos from configs.
        output_tensor_ids_op_name (dict): dictionary of output tensor op names.

    Returns:
        cfgs (dict).
    """
    for op_name in tune_cfg:
        inc_op_cfg = tune_cfg[op_name]
        for i, name in enumerate(op_name[0]):
            # to int8
            ipex_op_cfg = op_infos_from_cfgs[name]
            input_tensor_infos = ipex_op_cfg['input_tensor_infos']
            for index, input_tensor_info in enumerate(input_tensor_infos):
                if 'force_dtype' not in input_tensor_info.keys():
                    continue
                if input_tensor_info['force_dtype'] == 'torch.qint8' or \
                        input_tensor_info['force_dtype'] == 'torch.quint8':
                    # int8 -> int8
                    if inc_op_cfg['weight']['dtype'] == 'int8':
                        inc_scheme = inc_op_cfg['activation']['scheme']
                        inc_algorithm = inc_op_cfg['activation']['algorithm']
                        ipex_op_cfg['input_tensor_infos'] = input_tensor_infos
                        activation_observer = generate_activation_observer(inc_scheme,
                                                                           inc_algorithm)
                        ipex_op_cfg['activation_observer'] = activation_observer
                    # int8 -> fp32
                    else:
                        input_tensor_infos[index]['force_dtype'] = 'torch.float32'
                    # modify pre_op output inf_dtype
                    if i == 0:
                        input_tensor_id = input_tensor_info['id']
                        input_tensor_dtype = input_tensor_info['force_dtype']
                        if input_tensor_id in output_tensor_ids_op_name.keys():
                            pre_op_name = output_tensor_ids_op_name[input_tensor_id]
                            pre_op_module = pre_op_name[0][0]
                            pre_op_state = pre_op_name[0][1]
                            pre_op_index = pre_op_name[0][2]
                            pre_op_infos = cfgs[pre_op_module][pre_op_state][pre_op_index] 
                            pre_op_output_infos = pre_op_infos['output_tensor_infos']
                            for index, pre_op_output in enumerate(pre_op_output_infos):
                                if pre_op_output['id'] == input_tensor_id:
                                    pre_op_output_infos[index]['inf_dtype'] = input_tensor_dtype
                                else:
                                    print('Do not find the input id', input_tensor_id)
                            pre_op_infos['output_tensor_infos'] = pre_op_output_infos
                            cfgs[pre_op_module][pre_op_state][pre_op_index] = pre_op_infos
                        else:
                            print("Don't track the previous op name for ", name)
            cfgs[name[0]][name[1]][name[2]] = ipex_op_cfg
    return cfgs

def paser_cfgs(cfgs): # pragma: no cover
    """Parse configs.

    Args:
        cfgs (dict): the input configs.
        

    Returns:
        ops_name (list): list of op names.
        tune_cfg (dict): dictionary of quantization configuration.
        op_infos_from_cfgs (dict): op infos from configs.
        output_tensor_ids_op_name (dict): dictionary of output tensor op names.
    """
    ops_name = []
    layer_output_infos_ids = []
    op_infos_from_cfgs = {}
    # record input_tensor_id and op_name 
    #{"0": [(" ", "q_op_infos", "0"), (" ", "q_op_infos", "1")]}
    input_tensor_ids_op_name = {}
    output_tensor_ids_op_name = {}
    for module_key in cfgs.keys():
        for state in cfgs[module_key]:
            if state == "layer_output_infos":
                for index, op_info in enumerate(cfgs[module_key][state]):
                    name = (module_key, state, index)
                    ops_name.append(name)
                    layer_output_infos_ids.append(op_info['id'])
                    op_infos_from_cfgs[name] = op_info
                continue
            for op_cfg_id in cfgs[module_key][state].keys():
                op_info = cfgs[module_key][state][op_cfg_id]
                name = (module_key, state, op_cfg_id)
                if name not in ops_name:
                    ops_name.append(name)
                else:
                    assert False, \
                    "Please check IPEX int8 configure json whether have the same name ops"
                op_infos_from_cfgs[name] = op_info
                input_tensors = op_info['input_tensor_infos']
                for input_tensor in input_tensors:
                    if 'id' not in input_tensor.keys():
                        continue
                    else:
                        input_tensor_id = input_tensor['id']
                    if input_tensor_id not in input_tensor_ids_op_name.keys():
                        input_tensor_ids_op_name[input_tensor_id] = [name]
                    else:
                        input_tensor_ids_op_name[input_tensor_id].append(name)
                output_tensors = op_info['output_tensor_infos']
                for output_tensor in output_tensors:
                    if 'id' not in output_tensor.keys():
                        continue
                    else:
                        output_tensor_id = output_tensor['id']
                    if output_tensor_id not in output_tensor_ids_op_name.keys():
                        output_tensor_ids_op_name[output_tensor_id] = [name]
                    else:
                        output_tensor_ids_op_name[output_tensor_id].append(name)
    return ops_name, op_infos_from_cfgs, input_tensor_ids_op_name, output_tensor_ids_op_name

def get_quantizable_ops_from_cfgs(ops_name, op_infos_from_cfgs, input_tensor_ids_op_name): # pragma: no cover
    """Get quantizable ops from configs, combine fused ops as one op.

    Args:
        ops_name (list): list of op names.
        op_infos_from_cfgs (dict): op infos from configs.
        input_tensor_ids_op_name (dict): dictionary of input tensor op names.

    Returns:
        cfgs (dict).
    """
    quantizable_ops = []
    seen_ops = []
    for name in ops_name:
        start = True
        if name in seen_ops:
            continue
        elif name[1] not in ['q_op_infos']:
            continue
        else:
            # judge fuse ops the first op
            op_info = op_infos_from_cfgs[name]
            output_tensors = op_info['output_tensor_infos']
            input_tensors = op_info['input_tensor_infos']
            for input_tensor in input_tensors:
                if 'inf_dtype' not in input_tensor.keys():
                    continue
                if input_tensor['inf_dtype'] == torch.float32:
                    pre_op_name = input_tensor_ids_op_name[input_tensor["id"]]
                    if pre_op_name[1] in ['q_op_infos']:
                        print(pre_op_name, "is not the fuse ops first op.")
                        start = False
                        continue
            if not start:
                continue
            # add quantizable ops, include op and fuse ops.
            q_ops, stack = [],[(name,[])]
            while stack:
                cur_name, cur = stack.pop()
                seen_ops.append(cur_name)
                if cur_name[1] not in ['q_op_infos']:
                    q_ops.append(cur)
                    break
                op_info = op_infos_from_cfgs[cur_name]
                output_tensors = op_info['output_tensor_infos']
                for output_tensor in output_tensors:
                    if output_tensor['inf_dtype'] == 'torch.qint8' or \
                                    output_tensor['inf_dtype'] == 'torch.quint8':
                        q_ops.append(cur + [cur_name])
                        break
                    try:
                        next_op_names = input_tensor_ids_op_name[output_tensor['id']]
                        for next_op_name in next_op_names:
                            stack.append((next_op_name, cur + [cur_name]))
                    except:
                        next_op_name = None
                    if next_op_name is None:
                        q_ops.append(cur + [cur_name])
            for q_op in q_ops:
                quantizable_ops.append(q_op)
    return quantizable_ops

def auto_copy(module):  # pragma: no cover
    """Get an IPEX prepared model and return a fp32 model.

    Args:
        module (object): IPEX prepared model.

    Returns:
        fp32 model.
    """
    from intel_extension_for_pytorch.quantization._quantization_state import AutoQuantizationStateModuleDict
    def _nn_sequential_patched_forward(cls, x):
        for module in cls:
            if not isinstance(module, AutoQuantizationStateModuleDict):
                x = module(x)
        return x
    new_module = copy.deepcopy(module)
    if hasattr(new_module, '_qconf_summary'):
        del new_module._qconf_summary
    if hasattr(new_module, '_fqn_to_auto_quant_state_map'):
        del new_module._fqn_to_auto_quant_state_map
    if hasattr(new_module, 'q_config'):
        del new_module.q_config
    def convert_to_dispatch_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(CopyTensorProxy)  # type: ignore[arg-type]
        else:
            return x
    global_disable_torch_function_override = False
    class CopyTensorProxy(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            nonlocal global_disable_torch_function_override
            if (
                # global override means disable the override here
                global_disable_torch_function_override or
                # to prevent printing things from going into an infinite loop
                func == torch.Tensor.__repr__ or
                # we don't need to override getters in this framework
                func.__name__ == '__get__'
            ):
                return super().__torch_function__(func, types, args, kwargs)
            kwargs = kwargs if kwargs else {}
            output = super().__torch_function__(func, types, args, kwargs)
            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        CopyConvertTensorProxy)  # pylint: disable=E0602
                assert output is not NotImplemented
            return output
        def __repr__(self):
            return f'CopyTensorProxy({super().__repr__()})'
    cur_module = None
    module_stack : List[torch.nn.Module] = []  # pylint: disable=E0602
    assert len(module.__class__.__bases__) == 1
    class CopyDispatchModule(module.__class__.__bases__[0]):
        def __call__(self, *args, **kwargs):
            new_args = torch.fx.node.map_aggregate(args, convert_to_dispatch_proxy)
            new_kwargs = torch.fx.node.map_aggregate(kwargs, convert_to_dispatch_proxy)
            orig_module_call = torch.nn.Module.__call__
            orig_nn_sequential_forward = torch.nn.Sequential.forward
            def _patched_module_call(self, *args, **kwargs):
                nonlocal cur_module
                old_module = cur_module
                cur_module = self
                nonlocal global_disable_torch_function_override
                try:
                    parent_module = module_stack[-1] if len(module_stack) else None
                    module_stack.append(self)
                    output = orig_module_call(self, *args, **kwargs)
                    return output
                finally:
                    module_stack.pop()
                    cur_module = old_module
            torch.nn.Module.__call__ = _patched_module_call
            torch.nn.Sequential.forward = _nn_sequential_patched_forward  # type: ignore[assignment]
            try:
                output = super().__call__(*new_args, **new_kwargs)
                def unwrap_proxy(a):
                    if isinstance(a, CopyTensorProxy):
                        a.__class__ = torch.Tensor  # type: ignore[assignment]
                    return a
                output = torch.fx.node.map_aggregate(output, unwrap_proxy)
                return output
            finally:
                torch.nn.Module.__call__ = orig_module_call
                torch.nn.Sequential.forward = orig_nn_sequential_forward  # type: ignore[assignment]
    new_module.__class__ = CopyDispatchModule
    return new_module

def fetch_module(model, op_name):
    """Get module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split('.')
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    return module

def set_module(model, op_name, new_module):
    """Set module with a given op name.

    Args:
        model (object): the input model.
        op_name (str): name of op.
        new_module (object): the input model.

    Returns:
        module (object).
    """
    module = model
    name_list = op_name.split('.')
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)
    return module

def simple_inference(model, input):
    """Record model output tensor.

    Args:
        model (object): the input model.
        input (object).

    Returns:
        output (object).
    """
    with torch.no_grad():
        if type(input) is dict:
            output = model(**input)
        elif type(input) is tuple or type(input) is list:
            try:
                output = model(*input)
            except:
                output = model(input)
        else:
            output = model(input)
    return output

def get_example_input(dataloader, i=1):
    """Get the example input.

    Args:
        dataloader (object): calibration dataset.

    Returns:
        example_inp (object).
    """
    iter = 0
    try:
        for example_inp, label in dataloader:
            if iter == i:
                break
            else:
                iter += 1
    except:
        for example_inp in dataloader:
            if iter == i:
                break
            else:
                iter += 1
    return example_inp


def get_fallback_order(adaptor, fp32_model, dataloader, tune_cfg, 
                       confidence_batches, fallback=False, requantize_cfgs=None):
    """Get the fall back order for strategy.

    Args:
        fp32_model (object): the input model.
        dataloader(torch.utils.data.DataLoader): The calibration dataloader.
        tune_cfg (dict): dictionary of quantization configuration.
        confidence_batches (int): number of confidence batches.
        fallback (bool): if the order is fallback.

    Returns:
        ordered_ops (dict/list): The fallback order for strategy.
    """
    fp32_model.eval()
    order_dict = {}
    for i in range(0, confidence_batches):
        example_input = get_example_input(dataloader, i)
        if fallback:
            ordered_ops = get_mse_order_per_fp32(adaptor, fp32_model, example_input, tune_cfg)
            for i, name in enumerate(ordered_ops):
                order_dict[name] = order_dict.get(name, 0) + len(order_dict) - i
            ordered_ops = sorted(order_dict, key=lambda k: order_dict[k], reverse=True)
        else:
            ordered_ops = get_mse_order_per_int8(adaptor, fp32_model, example_input, tune_cfg)
            for i, name in enumerate(ordered_ops):
                order_dict[name] = order_dict.get(name, 0) + len(order_dict) - i
    return ordered_ops

op_cfg_mapping = {}
def get_mse_order_per_fp32(adaptor, model, example_inp, tune_cfg):
    """This is a helper method to check the mse influence to last module after QDQ(quant/dequant).

    Args:
        model (torch.fx.GraphModule/torch.nn.Module): A torch model.
        example_inp (object): example inputs.
        tune_cfg (dict): dictionary of quantization configuration.

    Returns:
        fallback_order (dict/list): The fallback order for strategy.
    """
    inner_output = None
    def output_hook(self, input, output):
        nonlocal inner_output
        inner_output = output
        return output

    op_type_dict = {}
    for k, v in tune_cfg['op'].keys():
        op_type_dict[k] = v

    from ..pytorch import _cfg_to_qconfig, _cfgs_to_fx_cfgs, PyTorch_FXAdaptor
    op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
    # insert hook to get output tesnor from last module
    last_module_name = list(op_cfgs.keys())[-1]
    module = fetch_module(model, last_module_name) # get last module
    module.register_forward_hook(output_hook)
    # record fp32 model output tensor at first
    output_fp32 = simple_inference(model, example_inp)
    inner_output_fp32 = inner_output

    fx_op_cfgs = {}
    fallback_order = {}
    logger.info('Evaluate the sensitivity for each int8 operation')
    for op_name, qconfig in tqdm(op_cfgs.items()):
        global op_cfg_mapping
        if op_name not in op_cfg_mapping:
            op_cfg_mapping[op_name] = qconfig
        tmp_model = copy.deepcopy(model)
        if not qconfig:
            continue
        op_cfgs[op_name] = None
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
        op_cfgs[op_name] = qconfig
        from torch.quantization.quantize_fx import prepare_fx,convert_fx
        # do quantization
        if adaptor.sub_module_list is None:
            if adaptor.version.release >= Version("1.13.0").release:  # pragma: no cover
                tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
            else:
                tmp_model = prepare_fx(tmp_model, fx_op_cfgs,)
        else:
            PyTorch_FXAdaptor.prepare_sub_graph(adaptor.sub_module_list, fx_op_cfgs, \
                                                tmp_model, prefix='')
        simple_inference(tmp_model, example_inp)
        if adaptor.sub_module_list is None:
            tmp_model = convert_fx(tmp_model)
        else:
            PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, \
                                                tmp_model, prefix='')

        # insert hook to get output tesnor from last module
        module = fetch_module(tmp_model, list(op_cfgs.keys())[-1]) # get last module
        module.register_forward_hook(output_hook)
        output_qdq = simple_inference(tmp_model, example_inp)
        inner_output_int8 = inner_output.dequantize() if \
          inner_output.dtype == torch.quint8 else inner_output
        mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
        fallback_order[(op_name, op_type_dict[op_name])] = mse_val

    ordered_ops = sorted(fallback_order.keys(), key=lambda key: fallback_order[key], \
                                    reverse=False)
    min_mse, max_mse = fallback_order[ordered_ops[0]], fallback_order[ordered_ops[-1]]

    if min_mse < 0.8 * max_mse:
        return ordered_ops


    double_check_list = []
    for op_name in ordered_ops:
        if min_mse <= fallback_order[op_name] <= (max_mse - min_mse) * 0.1 + min_mse:
            double_check_list.append(op_name)

    check_num = min(len(ordered_ops)//10, 5)
    double_check_list = ordered_ops[:check_num]
    worst_op_name = ordered_ops[-1]
    op_cfgs[worst_op_name[0]] = None # fallback worst module first
    new_fallback_order = {}

    logger.info('Evaluate the sensitivity gradient for selected operations')
    for op_name, op_type in tqdm(double_check_list):
        tmp_model = copy.deepcopy(model)
        qconfig = op_cfgs[op_name]
        op_cfgs[op_name] = None
        fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
        op_cfgs[op_name] = qconfig
        from torch.quantization.quantize_fx import prepare_fx,convert_fx
        # do quantization
        if adaptor.sub_module_list is None:
            if adaptor.version.release >= Version("1.13.0").release:  # pragma: no cover
                tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
            else:
                tmp_model = prepare_fx(tmp_model, fx_op_cfgs,)
        else:
            PyTorch_FXAdaptor.prepare_sub_graph(adaptor.sub_module_list, fx_op_cfgs, \
                                                tmp_model, prefix='')
        simple_inference(tmp_model, example_inp)
        if adaptor.sub_module_list is None:
            tmp_model = convert_fx(tmp_model)
        else:
            PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, \
                                                tmp_model, prefix='')

        # insert hook to get output tesnor from last module
        module = fetch_module(tmp_model, last_module_name) # get last module
        module.register_forward_hook(output_hook)
        output_qdq = simple_inference(tmp_model, example_inp)
        inner_output_int8 = inner_output.dequantize() if \
          inner_output.dtype == torch.quint8 else inner_output
        mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
        new_fallback_order[(op_name, op_type_dict[op_name])] = mse_val

    ordered_ops = sorted(new_fallback_order.keys(), key=lambda key: new_fallback_order[key], \
                                    reverse=False)

    return ordered_ops

def get_mse_order_per_int8(adaptor, fp32_model, example_input, tune_cfg):
    """This is a helper method to check the mse influence to last module after QDQ(quant/dequant).

    Args:
        model (torch.fx.GraphModule/torch.nn.Module): A torch model.
        example_inp (object): example inputs.
        tune_cfg (dict): dictionary of quantization configuration.
        
    Returns:
        fallback_order (dict/list): The fallback order for strategy.
    """
    inner_output = None
    def output_hook(self, input, output):
        nonlocal inner_output
        inner_output = output
        return output

    op_type_dict = {}
    for k, v in tune_cfg['op'].keys():
        op_type_dict[k] = v

    example_inp = example_input

    from ..pytorch import _cfg_to_qconfig
    op_cfgs = _cfg_to_qconfig(tune_cfg, tune_cfg["approach"])
    module = fetch_module(fp32_model, list(op_cfgs.keys())[-1]) # get last module
    # insert hook to get output tesnor from last module
    module.register_forward_hook(output_hook)
    # record fp32 model output tensor at first
    output_fp32 = simple_inference(fp32_model, example_inp)
    inner_output_fp32 = inner_output

    quant_list = []
    for k, v in tune_cfg['op'].items():
        if k[1] in ['LayerNorm', 'Dropout', 'InstanceNorm3d']:
            continue
        if v['weight']['dtype'] == 'fp32':
            quant_list.append(k)
    fallback_order = {}
    logger.info('Evaluate the sensitivity for each fp32 operation')
    for op_name, op_type in tqdm(quant_list):
        if op_name in op_cfg_mapping:
            tmp_model = copy.deepcopy(fp32_model)
            from ..pytorch import _cfg_to_qconfig, _cfgs_to_fx_cfgs, PyTorch_FXAdaptor
            op_cfgs[op_name] = op_cfg_mapping[op_name]
            fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, tune_cfg["approach"])
            from torch.quantization.quantize_fx import prepare_fx,convert_fx
            # do quantization
            if adaptor.sub_module_list is None:
                if adaptor.version.release >= Version("1.13.0").release:  # pragma: no cover
                    tmp_model = prepare_fx(tmp_model, fx_op_cfgs, example_inp)
                else:
                    tmp_model = prepare_fx(tmp_model, fx_op_cfgs,)
            else:
                PyTorch_FXAdaptor.prepare_sub_graph(adaptor.sub_module_list, fx_op_cfgs, \
                                                    tmp_model, prefix='')
            simple_inference(tmp_model, example_inp)
            if adaptor.sub_module_list is None:
                tmp_model = convert_fx(tmp_model)
            else:
                PyTorch_FXAdaptor.convert_sub_graph(adaptor.sub_module_list, \
                                                    tmp_model, prefix='')


            # record int8 model output tensor
            module = fetch_module(tmp_model, list(op_cfgs.keys())[-1]) # get last module
            module.register_forward_hook(output_hook)
            output_qdq = simple_inference(tmp_model, example_inp)
            inner_output_int8 = inner_output
            if inner_output_fp32.dtype == torch.quint8:
                inner_output_fp32 = inner_output_fp32.dequantize()
            if inner_output_int8.dtype == torch.quint8:
                inner_output_int8 = inner_output_int8.dequantize()

            mse_val = (inner_output_fp32 - inner_output_int8).pow(2).sum()
            fallback_order[(op_name, op_type_dict[op_name])] = mse_val
            # re-insert fp32 module into model
    ordered_ops = sorted(fallback_order.keys(), key=lambda key: fallback_order[key], \
                                            reverse=False)
    return ordered_ops

def get_torch_version():
    """Get torch version."""
    from packaging.version import Version
    try:
        torch_version = torch.__version__.split('+')[0]
    except ValueError as e:  # pragma: no cover
        assert False, 'Got an unknown version of torch: {}'.format(e)
    version = Version(torch_version)
    return version
