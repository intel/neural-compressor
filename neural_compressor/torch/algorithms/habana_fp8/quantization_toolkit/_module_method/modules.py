import torch
import torch.nn as nn
import torch.nn.functional as F

from habana_frameworks.torch.hpex.kernels.Fp8Ops import cast_from_fp8, cast_to_fp8_v2
from .._quant_common.quant_config import Fp8cfg
import math

config = Fp8cfg().cfg

# All tensors to be quantized are wrapped by QuantizedModule, which contains buffers with relevant information
# forward pass is MeasuredFunction, which quantizes and updates statistics
class QuantizedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.empty = torch.Tensor([0])
        self.scale = torch.Tensor([1])
        self.curr_scale = torch.Tensor([1])
        self.std = torch.Tensor([0])
        self.quant = False
        self.mse = []
        self.name = ''
        self.qtype = config['fp8_config']

    def forward(self, x):
        if self.quant:
            scale = self.scale
            stochastic = config['stochastic']
            scale = torch.Tensor([scale])
            qx,_ = cast_to_fp8_v2(x, scale, self.empty, stochastic, out_dtype=self.qtype)
            self.curr_scale = scale
            if config['fake_quant'] == True:
                qx = cast_from_fp8(qx, 1/scale, x.dtype)
            return qx
        else:
            return x
 
    def real_quant(self):
        return self.quant and (config['fake_quant'] == False)

    def start_quant(self):
        self.quant = True

    def get_curr_scale(self):
        return self.curr_scale

    def stop_quant(self):
        self.quant = False

    def set_scale(self, x):
        self.scale = x

    def set_name(self,name):
        self.name = name

    # std is normalized my max(x)
    def set_std(self, x):
        self.std = torch.std(x) / torch.max(x)

# Measured versions of used classes
class QuantizedBmm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8_x = QuantizedModule()
        self.fp8_y = QuantizedModule()

    def forward(self, x, y):
        if self.fp8_x.real_quant():
            qy = self.fp8_y(y) 
            qx = self.fp8_x(x)
            x_scale = self.fp8_x.get_curr_scale()
            y_scale = self.fp8_y.get_curr_scale()
            out = torch.ops.hpu.fp8_gemm_v2(qx, False, qy, False, None, x.dtype, 1/x_scale, 1/y_scale, None, None)
            return out
        else:
            qx = self.fp8_x(x)
            qy = self.fp8_y(y)
            return torch.bmm(qx, qy)

class QuantizedMatMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8_x = QuantizedModule()
        self.fp8_y = QuantizedModule()
        # self.fp8_out = MeasuredModule()

    def forward(self, x, y):
        # return self.fp8_out(torch.matmul(self.fp8_x(x), self.fp8_y(y)))
        if self.fp8_x.real_quant():
            x, y = preprocess_fp8(x, y)
            qy = self.fp8_y(y) 
            qx = self.fp8_x(x)
            x_scale = self.fp8_x.get_curr_scale()
            y_scale = self.fp8_y.get_curr_scale()
            return torch.ops.hpu.fp8_gemm_v2(qx, False, qy, False, None, x.dtype, 1/x_scale, 1/y_scale, None, None)
            
        else:
            qx = self.fp8_x(x)
            qy = self.fp8_y(y) 
            return torch.matmul(qx, qy)

class QuantizedLinear(nn.Linear):
    def __init__(self, current_module, *args, **kwargs):
        in_features = current_module.in_features
        out_features = current_module.out_features
        super().__init__(in_features=in_features, out_features=out_features, *args, **kwargs)
        self.weight = current_module.weight
        self.weight.data = self.weight.data #.transpose(0,1)
        self.bias = current_module.bias

        self.weight_quantizer = QuantizedModule()
        self.x_quantizer = QuantizedModule()
        self.weight_is_q = False

    def quantize_weights(self):
        self.weight.requires_grad = False
        self.weight.data = self.weight_quantizer(self.weight.data)
        self.weight_is_q = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.data 
        if self.x_quantizer.real_quant():
            input, weight = preprocess_fp8(input, weight)
            qx = self.x_quantizer(input)

            if not self.weight_is_q:
                qy = self.weight_quantizer(weight)
            else:
                qy = weight
            x_scale = self.x_quantizer.get_curr_scale()
            y_scale = self.weight_quantizer.get_curr_scale()

            out = torch.ops.hpu.fp8_gemm_v2(qx, False, qy, True, None, input.dtype, 1/x_scale, 1/y_scale, None, None)

        else:
            qx = self.x_quantizer(input)
            qy = self.weight_quantizer(weight)
            out = torch.matmul(qx, qy.t())
        
        if self.bias != None:
            return out + self.bias
        else:
            return out

if 'LoRACompatibleConv' in config['allowlist']['types']:
    try:
        from diffusers.models.lora import LoRACompatibleConv
        class QuantizedLoRACompatibleConv(LoRACompatibleConv):
            def __init__(self):
                super().__init__()

            def set_qmodule(self):
                self.weight_scale = 0
                self.fp8_kernel = QuantizedModule()
                self.fp8_x = QuantizedModule()
                self.weight_is_q = False

            def forward(self, hidden_states, scale: float = 1.0):
                if self.fp8_x.real_quant(): 
                    qx = self.fp8_x(hidden_states)
                    if not self.weight_is_q:
                        qweight = self.fp8_kernel(self.weight)
                        curr_weight_scale = self.fp8_kernel.get_curr_scale()
                    else:
                        qweight = self.weight
                        curr_weight_scale = self.weight_scale

                
                    out = torch.ops.hpu.conv2d_fp8(input=qx, weight=qweight, bias=None, stride=self.stride,\
                        padding=self.padding, dilation=self.dilation, groups=self.groups, out_dtype=hidden_states.dtype)
                    out /= (curr_weight_scale*self.fp8_x.get_curr_scale())
                    if self.bias != None:
                        out += torch.unsqueeze(torch.unsqueeze(self.bias,1), 1)
                
                    if self.lora_layer is not None:
                        out += scale * self.lora_layer(hidden_states)

                    return out

                else:
                    ## TODO - qdq in this setting doesnt quantize weights!
                    qx = self.fp8_x(hidden_states)
                    qweight = self.fp8_kernel(self.weight)
                    
                    return super(QuantizedLoRACompatibleConv, self).forward(qx, scale)
    except:
        print('no LoRACompatibleConv module found')


if 'LoRACompatibleLinear' in config['allowlist']['types']:
    try:
        from diffusers.models.lora import LoRACompatibleLinear
        class QuantizedLoRACompatibleLinear(LoRACompatibleLinear):
            def __init__(self):
                super().__init__()

            def set_qmodule(self):
                # TODO: same names in all modules
                self.weight_scale = 0
                self.weight_quantizer = QuantizedModule()
                self.x_quantizer = QuantizedModule()
                self.weight_is_q = False

            def forward(self, hidden_states, scale: float = 1.0):
                weight = self.weight.data 
                if self.x_quantizer.real_quant():
                    hidden_states, weight = preprocess_fp8(hidden_states, weight)
                    qhidden_states = self.x_quantizer(hidden_states)

                    if not self.weight_is_q:
                        qy = self.weight_quantizer(weight)
                    else:
                        qy = weight
                    x_scale = self.x_quantizer.get_curr_scale()
                    y_scale = self.weight_quantizer.get_curr_scale()
                    
                    out = torch.ops.hpu.fp8_gemm_v2(qhidden_states, False, qy, True, None, hidden_states.dtype, 1/x_scale, 1/y_scale, None, None)
                    if self.bias != None:
                        out = out + self.bias

                    if self.lora_layer is not None:
                        out = out  + (scale * self.lora_layer(hidden_states))
                
                # TODO: weight is not quantized in QDQ here
                else:
                    qx = self.x_quantizer(hidden_states)
                    qy = self.weight_quantizer(weight)
                    out = super(QuantizedLoRACompatibleLinear, self).forward(qx, scale)
                    
                return out
    except:
        print('no LoRACompatibleLinear module found')

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, current_module, *args, **kwargs):
        in_channels = current_module.in_channels
        out_channels = current_module.out_channels
        kernel_size = current_module.kernel_size
        stride = current_module.stride
        padding = current_module.padding
        dilation = current_module.dilation
        groups = current_module.groups
        has_bias = current_module.bias!=None
        padding_mode = current_module.padding_mode
        super().__init__(in_channels = in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, groups=groups,
                        bias=has_bias, padding_mode=padding_mode)
        self.fp8_kernel = QuantizedModule()
        self.fp8_x = QuantizedModule()
        self.weight = current_module.weight
        self.bias = current_module.bias
        self.weight_is_q = False
        self.weight_scale = 0

    def quantize_weights(self):
        self.weight.requires_grad = False
        self.weight.data = self.fp8_kernel(self.weight.data)
        self.weight_scale = self.fp8_kernel.get_curr_scale()
        self.weight_is_q = True

    def forward(self, x):
        if self.fp8_x.real_quant(): 
            qx = self.fp8_x(x)
            if not self.weight_is_q:
               qweight = self.fp8_kernel(self.weight)
               curr_weight_scale = self.fp8_kernel.get_curr_scale()
            else:
               qweight = self.weight
               curr_weight_scale = self.weight_scale

            out = torch.ops.hpu.conv2d_fp8(input=qx, weight=qweight, bias=None, stride=self.stride,\
                padding=self.padding, dilation=self.dilation, groups=self.groups, out_dtype=x.dtype)
            out /= (curr_weight_scale*self.fp8_x.get_curr_scale())
            if self.bias != None:
                out += torch.unsqueeze(torch.unsqueeze(self.bias,1), 1)
            
            return out

            # return torch.ops.hpu.conv2d_fp8(input=qx, weight=qweight, bias=self.bias, stride=self.stride,\
            #      padding=self.padding, dilation=self.dilation, groups=self.groups, out_dtype=x.dtype)

        else:
            return F.conv2d(input=self.fp8_x(x), weight=self.fp8_kernel(self.weight), 
                    bias=self.bias, stride=self.stride, 
                    padding=self.padding, dilation=self.dilation,
                    groups = self.groups)


def preprocess_fp8(input, weight):
    qx = input
    if qx.dim() < weight.dim():
        while qx.dim() != weight.dim():
            qx = torch.unsqueeze(qx, 0)
    if qx.dim() > weight.dim():
        while qx.dim() != weight.dim():
            weight = torch.unsqueeze(weight,0)

    return qx, weight

## TODO: below is an old implementation of MeasuredAdd and Measured GELU. Need to update
# class MeasuredAdd(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fp8_x = MeasuredModule(quantize=config['quantize'],
#                                     stats_shape=(1 + config['history_len'],))
#         self.fp8_y = MeasuredModule(quantize=config['quantize'],
#                                     stats_shape=(1 + config['history_len'],))
#         self.fp8_out = MeasuredModule(quantize=config['quantize'],
#                                       stats_shape=(1 + config['history_len'],))

#     def forward(self, x, y):
#         if config['fp8_add']:
#             return self.fp8_out(self.fp8_x(x) + self.fp8_y(y))
#         else:
#             return x + y

# class MeasuredGELU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fp8_x = MeasuredModule(quantize=config['quantize'],
#                                     stats_shape=(1 + config['history_len'],))
#         self.fp8_out = MeasuredModule(quantize=False,
#                                       stats_shape=(1 + config['history_len'],))

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         if config['fp8_gelu']:
#             return self.fp8_out(F.gelu(self.fp8_x(x)))
#         else:
#             return F.gelu(x)
