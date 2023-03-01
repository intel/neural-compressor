#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from neural_compressor.utils.utility import LazyImport

torch = LazyImport('torch')
from ...utils import logger


def model_forward(model, dataloader, iters):
    try:
        cnt = 0
        for idx, (input, label) in enumerate(dataloader):
            output = model(input)
            cnt += 1
            if cnt >= iters:
                break
    except Exception as e:
        cnt = 0
        for idx, input in enumerate(dataloader):
            if isinstance(input, dict):
                out = model(**input)
            else:
                out = model(input)
            cnt += 1
            if cnt >= iters:
                break


class TorchSmoothQuant:
    """
    Fake input channel quantization, for more details please refer to
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    Currently, we only handle the layers whose smooth scale could be absorbed, we will support other layers later.
    We only support inplace mode which means the model weights will be changed, you can call recover function only
    once to recover the weights if needed
    """

    def __init__(self, model, dataloader, traced_model=None):
        """
        :param model: Torch model :param dataloader: Calibration dataloader :param traced_model: A specific model
        shares the same architecture as the model and could be traced by torch.jit. If not supplied, we use model
        instead.
        """
        self.model = model
        device, dtype = self._get_device()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.dataloader = dataloader
        self.input_maxes = {}
        self.traced_model = traced_model
        if self.traced_model == None:
            self.traced_model = self.model
        self.weight_scale_info = {}
        self.absorb_scales_info = {}

    def _get_device(self):
        """
        Get the model device
        :return:Model device
        """
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def _get_module(self, key):
        """
        Get the module by the name parsed by jit
        :param key: the module name with the jit format
        :return: the module in original model
        """
        attrs = key.split('.')
        module = self.model
        for attr in attrs:
            try:
                attr = int(attr)
                module = module[attr]
            except:
                module = getattr(module, attr)
        return module

    def _save_input_pc_hook(self, name):
        """
        A forward hook to save input max of a module
        :param name: the module name
        :return: A hook function
        """

        def save_input_hook(module, inputs, outputs):
            if name not in self.input_maxes.keys():
                self.input_maxes[name] = []
            input = inputs[0]
            ##TODO check input channel is correct
            if len(module.weight.shape) == 4:  ##conv3d or conv1d not suppoted now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            self.input_maxes[name].append(max_tensor)

        return save_input_hook

    def _add_observer(self, modules):
        """
        :param modules: the modules which the observer will insert to
        :return:
        """
        self.hook_handles = []
        for key in modules.keys():
            hook_func = self._save_input_pc_hook(key)
            hook_handle = modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)

    def _remove_observer(self):
        """
        remove the observer from the model
        :return:
        """
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    # ##https://gist.github.com/sailfish009/28b54c8aa6398148a6358b8f03c0b611
    # def percentile(t: torch.tensor, q: float):
    #     """
    #     Return the ``q``-th percentile of the flattened input tensor's data.
    #
    #     CAUTION:
    #      * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
    #      * Values are not interpolated, which corresponds to
    #        ``numpy.percentile(..., interpolation="nearest")``.
    #
    #     :param t: Input tensor.
    #     :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    #     :return: Resulting value (scalar).
    #     """
    #     # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    #     # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    #     # so that ``round()`` returns an integer, even if q is a np.float32.
    #     k = 1 + round(.01 * float(q) * (t.numel() - 1))
    #     result = t.view(-1).kthvalue(k).values.item()
    #     return result

    def _calibrate(self, absorb_to_layer, calib_iter):
        """
        :param absorb_to_layer: A dict,key is the absorb layer, val is a list of the to be smoothed layer
        :param calib_iter: Data size for calibration
        :return: A dict that saved the layer name and the channe-wised max value info
        """
        layer_to_absorb = {}
        for key in absorb_to_layer:
            for layer_name in absorb_to_layer[key]:
                layer_to_absorb[layer_name] = key
        hook_module_names_tmp = [absorb_to_layer[key][0] for key in absorb_to_layer.keys()]
        hook_modules = {}

        for index, name in enumerate(hook_module_names_tmp):
            module = self._get_module(name)
            if isinstance(module, torch.nn.Linear) or isinstance(module,
                                                                 torch.nn.Conv2d):
                if isinstance(module, torch.nn.Conv2d):
                    if module.groups > 1 and module.in_channels == module.out_channels and \
                            module.groups == module.in_channels:
                        continue
                    else:
                        pass

                hook_modules[name] = module
        if len(hook_modules) == 0:
            return {}

        self._add_observer(hook_modules)
        self._dump_min_max(calib_iter=calib_iter)
        self._remove_observer()
        return self.input_maxes

    def _dump_min_max(self, calibration_method="min_max", calib_iter=100):
        """
        Dump min max per channel information, the min max value will be saved in input_maxes attribute
        :param calibration_method: only support min_max currently
        :param calib_iter: Sample size for calibration
        :return:
        """
        model_forward(self.model, self.dataloader, calib_iter)
        ##stack
        for key in self.input_maxes.keys():
            val = self.input_maxes[key]
            val = torch.stack(val, dim=0)
            val = torch.max(torch.abs(val), dim=0)[0]  ##FIXME should add abs
            self.input_maxes[key] = val

    def _reshape_in_channel_to_last(self, layer_name):
        """
        Move the input channel to the last dim
        :param layer_name: Layer name
        :return: The reshaped weight
        """
        weight = self._get_module(layer_name).weight  ##TODO oc*ic, support transposed conv
        if len(weight.shape) == 4:
            weight = weight.permute(0, 2, 3, 1)
            weight = weight.reshape(-1, weight.shape[-1])
        return weight

    def _scale_layer_weight(self, layer_name, scale):  ##input channel
        """
        Scale the layer weights at input channel
        :param layer_name: The layer name
        :param scale: The scale to be multiplied
        :return:
        """
        layer = self._get_module(layer_name)
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
            scale = scale.view(1, scale.shape[0], 1, 1)
            layer.weight *= scale
        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])
            layer.weight *= scale
        else:
            logger.warning(f"found unsupported layer {type(layer)}, try to multiply scale directly ")
            layer.weight *= scale

    def _absorb_scales(self, layer_name, scale):  ##output channel
        """
        Absorb the scale to the layer at output channel
        :param layer_name: The module name
        :param scale: The scale to be absorbed
        :return:
        """
        layer = self._get_module(layer_name)
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
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(
                    torch.ones(weight, requires_grad=False))
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(
                    bias, requires_grad=False)

        elif isinstance(layer, torch.nn.Conv2d):
            ##the order could not be changed
            if hasattr(layer, "bias") and (layer.bias != None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1 , 1, 1)
            layer.weight *= scale

        elif isinstance(layer, torch.nn.Linear):
            if hasattr(layer, "bias") and (layer.bias != None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1)
            layer.weight *= scale

        else:
            logger.warning(f"found unsupported layer {type(layer)}, try to multiply scale to weight and bias directly ")
            if hasattr(layer, "weight") and layer.weight != None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias != None:
                layer.bias *= scale

    def _adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5):
        """
        adjust the weights and biases
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight
        :return:
        """
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            input_max = absorb_to_input_maxes[key]
            layers = absorb_to_layer[key]
            weights = []
            for layer in layers:
                weight = self._reshape_in_channel_to_last(layer)
                weights.append(weight)

            weights = torch.cat(weights, dim=0)

            weight_max_per_channel = torch.max(torch.abs(weights), dim=0)[0]
            input_power = torch.pow(input_max, alpha)
            logger.debug(f"{max(input_max)}, {min(input_power)}")
            weight_power = torch.pow(weight_max_per_channel, 1 - alpha)
            #logger.info(f"{absorb_to_layer[key][0]} layer sparsity is
            # {1.0-torch.count_nonzero(input_power)/input_power.numel()}")

            scale = torch.clip(input_power / weight_power, min=1e-5)
            scale[input_power == 0] = 1.0

            self._absorb_scales(key, 1.0 / scale)
            absorb_scales_info[key] = 1.0 / scale
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                self._scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
        return weight_scales_info, absorb_scales_info

    def _check_same_hyperparameters(self, percentile, op_types,
                                  scales_per_op, calib_iter):
        """
        :param percentile:
        :param op_types:
        :param scales_per_op:
        :param calib_iter:
        :return:
        """
        if len(self.input_maxes) == 0:
            self.percentile = percentile
            self.op_types = op_types
            self.scales_per_op = scales_per_op
            self.calib_iter = calib_iter
            return False
        if self.percentile != percentile or self.op_types != op_types \
                or self.scales_per_op != scales_per_op or self.calib_iter != calib_iter:
            self.percentile = percentile
            self.op_types = op_types
            self.scales_per_op = scales_per_op
            self.calib_iter = calib_iter
            return False
        else:
            return True

    def transform(self, alpha=0.5, percentile=99.999, op_types=['Linear', 'Conv2d', 'ConvTranspose2d'],
                  scales_per_op=False, calib_iter=100):
        """
        The main entry of smooth quant
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, please refer
        to the paper for more details
        :param percentile: Not supported now
        :param op_types: The op typed to be smooth quantized
        :param scales_per_op: Not supported now
        :param calib_iter: Data size for calibration
        :return: A FP32 model with the same architecture as the orig model but with different weight which will be
        benefit to quantization
        """
        if not isinstance(self.model, torch.nn.Module):
            logger.warning("smooth quant is ignored since the model is not a torch module")
            return self.model
        matched = self._check_same_hyperparameters(percentile, op_types, scales_per_op, calib_iter)
        with torch.no_grad():
            input_maxes = self.input_maxes
            if matched == False:  ##avoid multiple calibaration during tuning if the only difference is alpha
                self.recover()
                self.absorb_to_layer, no_absorb_layers = self._trace(
                    op_types)  ##TODO we need to insert mul layer for no_absorb_layers later
                if self.absorb_to_layer == None and no_absorb_layers == None:
                    logger.warning("sorry, could not trace the model, smooth quant is ignored")
                    logger.warning("if you are using huggingface model,"
                                   "you could set torchscript to True "
                                   "when loading the model or set the return_dict to False")
                    return self.model

                input_maxes = self._calibrate(self.absorb_to_layer, calib_iter)

            self.recover()
            self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(self.absorb_to_layer, input_maxes,
                                                                                      alpha)
            return self.model

    def recover(self):
        """
        recover the model weights
        :return:
        """
        with torch.no_grad():
            for key in self.weight_scale_info:
                self._scale_layer_weight(key, 1.0 / self.weight_scale_info[key])
            for key in self.absorb_scales_info:
                self._absorb_scales(key, 1.0 / self.absorb_scales_info[key])
            self.weight_scale_info = {} ##clear the data
            self.absorb_scales_info = {}

    def _trace(self, op_types):
        """
        Try the model to find the layers which can be smooth quantized.
        :param op_types: The op types to be smooth quantized
        :return:
        absorb_to_layer: A dict, absorb layer name:layers to be smooth quantized
        no_absorb_layers: A list saving the layers which could not find the absorb layer
        """
        tg = GraphTrace()
        for idx, input in enumerate(self.dataloader):
            example_inputs = input
            break
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.traced_model, example_inputs, op_types)
        return absorb_to_layer, no_absorb_layers


def get_parent(node):
    if node.inputs() == None:
        return None
    return list(node.inputs())[0].node()


def get_module(model, key):
    attrs = key.split('.')
    module = model
    for attr in attrs:
        try:
            attr = int(attr)
            module = module[attr]
        except:
            module = getattr(module, attr)
    return module


class GraphTrace:
    """
    """

    def __init__(self):
        self.supported_torch_module_to_aten = {
            "Linear": "aten::linear",
            "Conv2d": "aten::_convolution",
            "ConvTranspose2d": "aten::_convolution",
            "LayerNorm": "aten::layer_norm",
            "BatchNorm2d": "aten::batch_norm",
            "GroupNorm": "aten::group_norm",
            "InstanceNorm2d": "instance_norm",
        }
        ##TODO, must statisfy af(x)=f(ax),current skip layer may be incomplete
        self.skip_ops_to_find_absorb = ["aten::to",
                                        "aten::relu",
                                        "aten::leaky_relu",
                                        "aten::hardtanh"
                                        ]

        self.could_absorb_layers = ["aten::layer_norm", "aten::batch_norm", "aten::linear", "aten::_convolution",
                                    "aten::group_norm",
                                    "aten::instance_norm"]  ##TODO,suppport more norm

    def trace(self, model, dummy_input):
        traced_model = None
        optimize_numerics = False
        if isinstance(dummy_input, dict):
            try:
                traced_model = torch.jit.trace(model, dummy_input["input_ids"], strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                pass
        else:
            try:
                traced_model = torch.jit.trace(model, dummy_input, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                try:
                    traced_model = torch.jit.trace(model, dummy_input[0], strict=False)
                    traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
                except:
                    pass
        return traced_model

    def get_nodes(self, traced_model, op_types=['Linear']):
        if isinstance(op_types, str):
            op_types = [op_types]
        nodes = []
        for node in traced_model.graph.nodes():
            node_type = node.kind()
            for op_type in op_types:
                if node_type == op_type:
                    nodes.append((node, op_type))
                    break
        return nodes

    def get_prev_absorb_layer(self, nodes):
        prev_absorb_layer = []
        for node in nodes:
            parent = get_parent(node)
            while 1:
                if parent.kind() in self.skip_ops_to_find_absorb:
                    parent = get_parent(parent)
                    continue
                if parent.kind() in self.could_absorb_layers:
                    prev_absorb_layer.append(parent)
                else:
                    prev_absorb_layer.append(None)
                break
        return prev_absorb_layer

    def mapping_torch_module_to_aten(self, op_types):
        res = []
        for op in op_types:
            if op not in self.supported_torch_module_to_aten.keys():
                logger.warning(f"{op} is not supported in smooth quant, ignoring...")
                continue
            res.append(self.supported_torch_module_to_aten[op])
        res = list(set(res))
        return res

    def get_absorb_to_layer(self, model, example_input, op_types):
        traced_model = self.trace(model, example_input)
        if traced_model == None:
            return None, None
        aten_op_types = self.mapping_torch_module_to_aten(op_types)
        nodes_types = self.get_nodes(traced_model, aten_op_types)
        nodes = [node_type[0] for node_type in nodes_types]
        nodes_prev_absorb = self.get_prev_absorb_layer(nodes)
        absorb_to_layer = {}
        no_absorb_layers = []
        for index, absorb in enumerate(nodes_prev_absorb):
            if absorb == None:
                no_absorb_layers.append(nodes[index])
                continue
            node = nodes[index]
            layer_name = '.'.join(node.scopeName().split('/')[-1].split('.')[1:])
            absorb_name = '.'.join(absorb.scopeName().split('/')[-1].split('.')[1:])
            if absorb_name in absorb_to_layer.keys():
                absorb_to_layer[absorb_name].append(layer_name)
            else:
                absorb_to_layer[absorb_name] = [layer_name]
        absorb_to_layer = self.remove_unsupported_layers(model, absorb_to_layer)
        return absorb_to_layer, no_absorb_layers

    def remove_unsupported_layers(self, model, absorb_to_layer):
        res = {}

        for key in absorb_to_layer.keys():

            absorb_layer = get_module(model, key)
            layer_type = absorb_layer.__class__.__name__
            if layer_type not in self.supported_torch_module_to_aten.keys():
                continue
            supported = True
            for layer_name in absorb_to_layer[key]:
                layer = get_module(model, layer_name)
                layer_type = layer.__class__.__name__
                if layer_type not in self.supported_torch_module_to_aten.keys():
                    supported = False
                    break
            if supported:
                res[key] = absorb_to_layer[key]
        return res
