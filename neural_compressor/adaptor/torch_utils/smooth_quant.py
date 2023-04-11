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
            elif isinstance(input, list) or isinstance(input, tuple):
                out = model(*input)
            else:
                out = model(input)
            cnt += 1
            if cnt >= iters:
                break


def quant_dequant_w(m, num_bits=8, scheme='asym'):  ##TODO take sym as default
    if isinstance(m, torch.nn.Linear):
        x = m.weight
        if scheme == 'sym':
            q_min, q_max = -2. ** (num_bits - 1), 2. ** (num_bits - 1) - 1.
            scale = torch.abs(torch.max(x, dim=1).values) / (2 ** (num_bits - 1) - 1)
        else:
            q_min, q_max = 0, 2. ** num_bits - 1.
            scale = (torch.max(x, dim=1).values - torch.min(x, dim=1).values) / (2 ** num_bits - 1)
        scale = torch.clip(scale, min=1e-5)

        if scheme == 'sym':
            bias = 0
        else:
            bias = torch.round(0 - (torch.min(x, dim=1).values) / scale)
            bias = bias.unsqueeze(dim=-1)
        scale = scale.unsqueeze(dim=-1)
        q_x = x / scale + bias
        q_x.clamp_(q_min, q_max).round_()
        return (q_x - bias) * scale
    elif isinstance(m, torch.nn.Conv2d):
        x = m.weight
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(-1, x.shape[-1])
        if scheme == 'sym':
            q_min, q_max = -2. ** (num_bits - 1), 2. ** (num_bits - 1) - 1.
            scale = torch.abs(torch.max(x, dim=0).values) / (2 ** (num_bits - 1) - 1)
        else:
            q_min, q_max = 0, 2. ** num_bits - 1.
            scale = (torch.max(x, dim=0).values - torch.min(x, dim=0).values) / (2 ** num_bits - 1)
        scale = torch.clip(scale, min=1e-5)
        if scheme == 'sym':
            bias = 0
        else:
            bias = torch.round(0 - (torch.min(x, dim=0).values) / scale)
            bias = bias.unsqueeze(dim=0)
        scale = scale.unsqueeze(dim=0)

        q_x = x / scale + bias
        q_x.clamp_(q_min, q_max).round_()
        q_dq_x = (q_x - bias) * scale
        q_dq_x = q_dq_x.view(m.weight.shape[0], m.weight.shape[2], m.weight.shape[3], m.weight.shape[1])
        q_dq_x = torch.permute(q_dq_x, (0, 3, 1, 2))
        return q_dq_x
    else:
        logger.warning("unsupported layer type, please have a check")


def quant_dequant_x(x, num_bits=8):
    q_min, q_max = 0, 2. ** num_bits - 1.
    scale = (torch.max(x) - torch.min(x)) / (2 ** num_bits - 1)
    scale = torch.clip(scale, min=1e-5)
    bias = torch.round(0 - (torch.min(x)) / scale)
    q_x = x / scale + bias
    q_x.clamp_(q_min, q_max).round_()
    return scale * (q_x - bias)


class TorchSmoothQuant:
    """
    Fake input channel quantization, for more details please refer to
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    Currently, we only handle the layers whose smooth scale could be absorbed, we will support other layers later.
    We only support inplace mode which means the model weights will be changed, you can call recover function
    to recover the weights if needed
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
        self.input_values = {}
        self.output_values = {}
        self.input_maxes = {}
        self.hook_layer_names = []
        self.hook_values_handles = []
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
            if len(module.weight.shape) == 4:  ##conv3d or conv1d not supported now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            self.input_maxes[name].append(max_tensor)
            # self.input_values[name] = input
            # self.output_values[name] = outputs

        return save_input_hook

    def _save_input_output_hook(self, name):
        """
        A forward hook to save input and output values of a module
            param name: the module name
            return: A hook function
        """

        def save_input_output_hook(module, inputs, outputs):
            input = inputs[0]
            # if name in self.input_values:
            #     self.input_values[name].append(input)
            #     self.output_values[name].append(outputs)
            # else:
            self.input_values[name] = [input]  ##TODO save more,like 8
            self.output_values[name] = [outputs]  ##TODO do not save output

        return save_input_output_hook

    def _add_observer(self, modules, input_output_modules=None):
        """
        :param modules: the modules which the observer will insert to
        :return:
        """
        self.hook_handles = []
        for key in modules.keys():
            hook_func = self._save_input_pc_hook(key)
            hook_handle = modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)
        if input_output_modules:
            for key in input_output_modules.keys():
                hook_func = self._save_input_output_hook(key)
                hook_handle = input_output_modules[key].register_forward_hook(hook_func)
                self.hook_values_handles.append(hook_handle)

    def _remove_observer(self):
        """
        remove the observer from the model
        :return:
        """
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        if self.hook_values_handles:
            for hook_handle in self.hook_values_handles:
                hook_handle.remove()

    def _calibrate(self, absorb_to_layer, calib_iter, save_input_output=False):
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
                    if self._check_dw_conv(module):
                        pass
                    elif module.groups > 1:
                        continue

                hook_modules[name] = module
        if len(hook_modules) == 0:
            return {}
        hook_modules_input_output = {}
        for name in self.hook_layer_names:
            hook_modules_input_output[name] = self._get_module(name)
        self._add_observer(hook_modules, hook_modules_input_output)
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
        for key in self.input_values.keys():
            self.input_values[key] = torch.cat(self.input_values[key], dim=0)  ##this may introduce memory issue
            self.output_values[key] = torch.cat(self.output_values[key], dim=0)

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

    def _reshape_scale_for_weight(self, layer, scale):
        """
        reshape the scale for weight input channel, depthwise output channel
        :param layer:  torch module
        :param scale: orig scale
        :return: reshaped scale
        """
        if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:  ##only depthwise conv could hit here
            scale = scale.view(scale.shape[0], 1, 1, 1)  ##mount on output channel

        elif isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def _reshape_scale_for_input(self, layer, scale):
        """
        reshape the scale for input feature in channel
        :param layer:
        :param scale:
        :return:
        """
        if isinstance(layer, torch.nn.Conv2d):
            scale = scale.view(1, scale.shape[0], 1, 1)

        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])

        return scale

    def _scale_layer_weight(self, layer_name, scale):  ##input channel
        """
        Scale the layer weights at input channel, depthwise conv output channel
        :param layer_name: The layer name
        :param scale: The scale to be multiplied
        :return:
        """
        layer = self._get_module(layer_name)
        scale = self._reshape_scale_for_weight(layer, scale)
        layer.weight *= scale
        return scale

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
            scale = scale.view(scale.shape[0], 1, 1, 1)
            layer.weight *= scale

        elif isinstance(layer, torch.nn.Linear):
            if hasattr(layer, "bias") and (layer.bias != None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0], 1)
            layer.weight *= scale

        elif layer.__class__.__name__ == "LlamaRMSNorm" or layer.__class__.__name__ == "T5LayerNorm":  ##quite tricky
            layer.weight *= scale


        else:
            logger.warning(f"found unsupported layer {type(layer)}, try to multiply scale to weight and bias directly, "
                           f"this may introduce accuracy issue, please have a check ")
            if hasattr(layer, "weight") and layer.weight != None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias != None:
                layer.bias *= scale

    def _adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5):
        """
        adjust the weights and biases
        :param absorb_to_layer: A dict mapping absorb layer to smooth quantized layer
        :param input_maxes: The channel-wise input max info for layers
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, a float of a dict
        :return:
        """
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            if isinstance(alpha, float):
                alpha_key = alpha
            elif isinstance(alpha, dict):
                alpha_key = alpha[key]
            input_max = absorb_to_input_maxes[key]
            layers = absorb_to_layer[key]
            weights = []
            for layer in layers:
                weight = self._reshape_in_channel_to_last(layer)
                weights.append(weight)

            weights = torch.cat(weights, dim=0)

            weight_max_per_channel = torch.max(torch.abs(weights), dim=0)[0]
            input_power = torch.pow(input_max, alpha_key)
            logger.debug(f"{max(input_max)}, {min(input_max)}")
            weight_power = torch.pow(weight_max_per_channel, 1 - alpha_key)
            # logger.info(f"{absorb_to_layer[key][0]} layer sparsity is
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

    def _check_need_calibration(self, alpha, percentile, op_types,
                                scales_per_op, calib_iter):
        """
        check need calibration or not
        :param alpha: current alpha
        :param percentile: current percentile
        :param op_types: current op_types
        :param scales_per_op: current scales_per_op
        :param calib_iter:: current scales_per_op
        :return:
        """
        need_calib = True
        if len(self.input_maxes) == 0:  ## the first time
            need_calib = True
            self.alpha = alpha
            self.percentile = percentile
            self.op_types = op_types
            self.scales_per_op = scales_per_op
            self.calib_iter = calib_iter
            return need_calib

        if self.percentile == percentile and self.op_types == op_types \
                and self.scales_per_op == scales_per_op and self.calib_iter != calib_iter:
            if isinstance(alpha, float):
                need_calib = False
            elif self.alpha == "auto":
                need_calib = False

        self.alpha = alpha
        self.percentile = percentile
        self.op_types = op_types
        self.scales_per_op = scales_per_op
        self.calib_iter = calib_iter
        return need_calib

    def _check_dw_conv(self, module):
        if not isinstance(module, torch.nn.Conv2d):
            return False

        return module.groups > 1 and module.in_channels == module.out_channels and \
               module.groups == module.in_channels

    def _auto_tune_alpha(self, input_maxes, alpha_min=0.3, alpha_max=0.7, alpha_step=0.05, attn_method='min'):
        """
        Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly.
        input_maxes:
        alpha_min: min value of alpha search space.
        alpha_max: max value of alpha search space.
        alpha_step: step size of alpha search space.
        attn_method: criterion method used on attention ops; currently min, max and mean are supported.
        """
        logger.info("auto tuning alpha")
        import copy
        alpha_scale = 100
        alpha_space = list(range(round(alpha_min * alpha_scale), round((alpha_max + alpha_step) * alpha_scale),
                                 round(alpha_step * alpha_scale)))
        alpha_space = [alpha / alpha_scale for alpha in alpha_space]

        ans_layer2absorb, self.layer_to_absorb, ans = {}, {}, {}
        ## Searching optimal alphas
        for idx, key in enumerate(self.absorb_to_layer):
            absorb_to_layer_sample, input_max_op = {}, {}
            absorb_key = key
            absorb_to_layer_sample[absorb_key] = self.absorb_to_layer[absorb_key]
            loss_all_layers = {}
            for layer_key in self.absorb_to_layer[absorb_key]:
                # if self._check_dw_conv(get_module(self.model,layer_key)):

                if layer_key not in self.layer_to_absorb.values():
                    if layer_key in input_maxes:
                        self.layer_to_absorb[absorb_key] = layer_key
                layer_key_ = self.layer_to_absorb[absorb_key]
                input_max_op[layer_key] = input_maxes[layer_key_]
                loss_alpha = {}
                for alpha in alpha_space:
                    self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(absorb_to_layer_sample,
                                                                                              input_max_op, alpha)
                    input_of_op, output_of_op = self.input_values[layer_key], self.output_values[layer_key]
                    input_scale = self._reshape_scale_for_input(self._get_module(layer_key),
                                                                self.absorb_scales_info[absorb_key])
                    input_of_op_q = quant_dequant_x(input_of_op * input_scale)
                    layer = self._get_module(layer_key)
                    weight_qdq = quant_dequant_w(layer)
                    layer_cp = copy.deepcopy(layer)
                    layer_cp.weight.data = weight_qdq
                    output_of_op_q = layer_cp(input_of_op_q)
                    self.recover()
                    loss = torch.sum(torch.abs(output_of_op - output_of_op_q) ** 2)
                    loss_alpha[alpha] = loss
                    if layer_key not in ans:  # Update alpha results
                        ans[layer_key] = alpha
                    else:
                        ans[layer_key] = alpha if loss < loss_alpha[ans[layer_key]] else ans[layer_key]
                loss_all_layers[layer_key] = loss_alpha
                if absorb_key not in ans_layer2absorb:
                    ans_layer2absorb[absorb_key] = ans[layer_key]
                else:
                    if attn_method == 'max':
                        ans_layer2absorb[absorb_key] = max(ans_layer2absorb[absorb_key], ans[layer_key])
                    elif attn_method == 'min':
                        ans_layer2absorb[absorb_key] = min(ans_layer2absorb[absorb_key], ans[layer_key])
                    elif attn_method == 'mean':
                        pass
            if attn_method == 'mean':
                mean_loss = {}
                for alpha in alpha_space:
                    mean_loss[alpha] = 0
                    for key in loss_all_layers.keys():
                        mean_loss[alpha] += loss_all_layers[key][alpha]
                min_alpha = min(mean_loss, key=mean_loss.get)
                if len(loss_all_layers) > 1:
                    ans_layer2absorb[absorb_key] = min_alpha
        logger.info("auto tuning alpha done")
        return ans_layer2absorb

    def transform(self, alpha=0.5, percentile=99.999, op_types=['Linear', 'Conv2d'],
                  scales_per_op=False, calib_iter=100,
                  auto_alpha_args={'alpha_min': 0.3, 'alpha_max': 0.7, 'alpha_step': 0.05, 'attn_method': 'min'}):
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
        if isinstance(alpha, float) and (alpha < 0 or alpha > 1):
            logger.warning("alpha should be a float value in [0, 1] or 'auto' ")
            if alpha < 0:
                alpha = 0
                logger.warning("reset alpha to 0 ")
            if alpha > 1.0:
                alpha = 1.0
                logger.warning("reset alpha to 1.0 ")

        if not isinstance(self.model, torch.nn.Module):
            logger.warning("smooth quant is ignored since the model is not a torch module")
            return self.model
        self.recover()
        need_calibration = self._check_need_calibration(alpha, percentile, op_types, scales_per_op, calib_iter)
        with torch.no_grad():
            input_maxes = self.input_maxes
            if need_calibration:  ##avoid multiple calibaration during tuning if the only difference is alpha
                self.absorb_to_layer, no_absorb_layers = self._trace(
                    op_types)  ##TODO we need to insert mul layer for no_absorb_layers later
                for key in self.absorb_to_layer:
                    self.hook_layer_names += self.absorb_to_layer[key]
                if self.absorb_to_layer == None and no_absorb_layers == None:
                    logger.warning("sorry, could not trace the model, smooth quant is ignored")
                    logger.warning("if you are using huggingface model,"
                                   "you could set torchscript to True ")
                    return self.model
                save_input_output = False
                if alpha == "auto":
                    save_input_output = True

                input_maxes = self._calibrate(self.absorb_to_layer, calib_iter, save_input_output)
                if alpha == 'auto':
                    self.alpha_per_layer = self._auto_tune_alpha(input_maxes, **auto_alpha_args)  ##save the alpha

            if alpha == 'auto':
                alpha = self.alpha_per_layer

            self.weight_scale_info, self.absorb_scales_info = self._adjust_parameters(self.absorb_to_layer,
                                                                                      input_maxes, alpha)
            self.input_values, self.output_values = {}, {}
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
            self.weight_scale_info = {}  ##clear the data
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
            "InstanceNorm2d": "aten::instance_norm",
            "LlamaRMSNorm": "aten::mul",
            "T5LayerNorm": "aten::mul",
        }
        ##TODO, must statisfy af(x)=f(ax),current skip layer may be incomplete
        self.skip_ops_to_find_absorb = ["aten::to",
                                        "aten::relu",
                                        "aten::leaky_relu",
                                        "aten::hardtanh"
                                        ]

        self.could_absorb_layers = ["aten::layer_norm", "aten::batch_norm", "aten::linear", "aten::_convolution",
                                    "aten::group_norm",
                                    "aten::instance_norm",
                                    "aten::mul"]  ##TODO,suppport more norm

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
