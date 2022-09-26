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

import copy
import re
import numpy as np
from collections import UserDict
from ...utils.utility import LazyImport, CpuInfo

torch = LazyImport("torch")

def get_embedding_contiguous(model):
    """This is a helper function for nn.Embedding,
        and it will get input contiguous.

    Args:
        model (object): input model

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


def collate_torch_preds(results):
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
    if isinstance(input, dict) or isinstance(input, UserDict):
        output = tuple(input.values())
    elif isinstance(input, list) or isinstance(input, tuple):
        output = tuple(input)
    else:
        output = input
    return output


def append_attr(fx_model, model):
    """a helper method to append attributes for the symbolic traced model.

    Args:
        fx_model(torch.fx.GraphModule): The symbolic traced model.
        model(torch.nn.Module): The original model.

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
    # combine fuse ops as one op.
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
    # module: IPEX prepared model
    # return fp32 model
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
