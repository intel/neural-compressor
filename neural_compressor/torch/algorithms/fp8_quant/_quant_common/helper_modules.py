import torch.nn as nn
import torch
import math
from .quant_config import QuantMode, get_hqt_config, set_hqt_config

class BMM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)

class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.matmul(*args, **kwargs)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()

class Softmax(nn.Module):
      def __init__(self):
        super().__init__()

      def forward(self, x, dim = None):
        return torch.softmax(x, dim)

def quant_dequant(
        x, scale_quant_fcn=None, quant_fcn=None, scale_dequant_fcn=None, dequant_fcn=None
):
  y = x if (scale_quant_fcn is None) else scale_quant_fcn(x)
  y = y if (quant_fcn is None) else quant_fcn(y)
  y = y if (dequant_fcn is None) else dequant_fcn(y)
  y = y if (scale_dequant_fcn is None) else scale_dequant_fcn(y)
  return y

def matmul_fp8(input, other, out=None, out_dtype=torch.bfloat16, scale_input_inv=None, scale_other_inv=None):
  res = torch.ops.hpu.fp8_gemm_v2(input, False, other, False, out, out_dtype, scale_input_inv, scale_other_inv, None, False)
  return res

def measure_input(input, observer):
  for i in range(len(observer)):
    observer[i].measure(input[i])

def measure_output(output, observer):
  if observer:
    observer.measure(output)

def conv2d_fp8(input, other, bias,  stride, padding, dilation, groups, out_dtype=torch.bfloat16, scale_input_inv=None, scale_other_inv=None):
  return torch.ops.hpu.conv2d_fp8(input=input, weight=other, bias=bias, stride=stride,\
                        padding=padding, dilation=dilation, groups=groups, out_dtype=out_dtype, scale_input=scale_input_inv, scale_weight=scale_other_inv)

def extra_representation(org_repr, org_name, curr_repr):
  repr = f"original={org_name}," + (" "+ org_repr + "," if org_repr != "" else "")
  return f'{repr} {curr_repr}'

def _raise_lora_layer_error(layer_class):
    raise RuntimeError(f"{layer_class} quantization is not supported in case of lora_layer member is not None."
                       f" Can add {layer_class} to 'blocklist' field in quantization config file")

class PatchedMatmul(nn.Module):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      self.quant_input_0 = self._mod_config.inputs[0]
      self.quant_input_1 = self._mod_config.inputs[1]
      self.scale_input = nn.Parameter(mod_config.scale.inputs[0])
      self.scale_other = nn.Parameter(mod_config.scale.inputs[1])
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward(self, input, other):
    qinput = self.quant_input_0(input)
    qother = self.quant_input_1(other)
    output = matmul_fp8(qinput, qother, out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_other)
    return output

  def forward_measure(self, input, other):
    measure_input((input, other), observer = self._mod_config.inputs)
    output = self.forward_orig(input, other)
    measure_output(output, self._mod_config.outputs)
    return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_other dtype={self.scale_other.dtype}')

class PatchedLinear(nn.Module):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      # When offloading weights to disk using device_map, the module forward is overridden.
      # __dict__.update call again overrides the PatchedLinear forward with the forward that device_map planted.
      # So need to set PatchedLinear forawrd to be the right forward.
      self.forward = self.forward_quant
      self.quant_input_0 = self._mod_config.inputs[0]
      self.weight = nn.Parameter(self.weight.t().contiguous())
      self.scale_input = nn.Parameter(mod_config.scale.inputs[0])
      if isinstance(mod_config.scale.params['weight'], (torch.Tensor, float)):
        self.scale_weight = nn.Parameter(mod_config.scale.params['weight'])
      elif isinstance(mod_config.scale.params['weight'], dict):
        # PCQ weight is calculated with actual weight [0] and ones [1]
        self.scale_weight = nn.Parameter(mod_config.scale.params['weight'][0])
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward_quant(self, input):
    qinput = self._mod_config.inputs[0](input)
    y = matmul_fp8(qinput, self.weight, out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
    output = y+self.bias if (self.bias is not None) else y
    return output

  def forward_measure(self, input):
    measure_input((input,), observer = self._mod_config.inputs)
    output = self.forward_orig(input)
    measure_output(output, self._mod_config.outputs)
    return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_weight dtype={self.scale_weight.dtype}')

class PatchedLinearAllReduce(nn.Module):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.scoped_version =  mod.__class__.__name__ == "ScopedLinearAllReduce"
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      self.quant_input_0 = self._mod_config.inputs[0]
      self.quant_output = self._mod_config.outputs
      self.weight = nn.Parameter(self.weight.t().contiguous())
      self.scale_input = nn.Parameter(mod_config.scale.inputs[0])
      if isinstance(mod_config.scale.params['weight'], (torch.Tensor, float)):
        self.scale_weight = nn.Parameter(mod_config.scale.params['weight'])
      elif isinstance(mod_config.scale.params['weight'], dict):
        # PCQ weight is calculated with actual weight [0] and ones [1]
        self.scale_weight = nn.Parameter(mod_config.scale.params['weight'][0])
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward(self, input):
    # pre_all_reduce
    qinput = self.quant_input_0(input)
    output = matmul_fp8(qinput, self.weight, out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
    dqoutput = self.quant_output(output)
    if not self.scoped_version:
      self.all_reduce(dqoutput)
      dqoutput = self.post_all_reduce(dqoutput)
    return dqoutput

  def forward_measure(self, input):
    measure_input((input,), observer=self._mod_config.inputs)
    output = torch.matmul(input, self.weight.transpose(-1, -2))
    measure_output(output, self._mod_config.outputs)
    # in scoped version all reduce is being called outside of the layer
    if not self.scoped_version:
      self.all_reduce(output)
      output = self.post_all_reduce(output)
    return output

  def all_reduce(self, input):
    if self.mp_group is not None:
        from deepspeed import comm as dist
        dist.inference_all_reduce(input, group=self.mp_group)

  def post_all_reduce(self, input):
      output = input + self.bias if (self.bias is not None) else input
      return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_weight dtype={self.scale_weight.dtype}')

class PatchedLmHeadLinearAllreduce(nn.Module):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      self.quant_input_0 = self._mod_config.inputs[0]
      self.quant_output = self._mod_config.outputs
      self.weight = nn.Parameter(self.weight.t().contiguous())
      self.scale_input = nn.Parameter(mod_config.scale.inputs[0])
      if isinstance(mod_config.scale.params['weight'], (torch.Tensor, float)):
        self.scale_weight = nn.Parameter(mod_config.scale.params['weight'])
      elif isinstance(mod_config.scale.params['weight'], dict):
        # PCQ weight is calculated with actual weight [0] and ones [1]
        self.scale_weight = nn.Parameter(mod_config.scale.params['weight'][0])
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward(self, input):
      assert input.shape[
          -1] % self.world_size == 0, 'Please ensure that self.world_size is divisible by input.shape[-1]'
      input_shard = input.shape[-1] // self.world_size
      splittedInput = input[:, :, self.rank * input_shard:(self.rank + 1) * input_shard]
      qinput = self.quant_input_0(splittedInput)
      output = matmul_fp8(qinput, self.weight, out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
      dqoutput = self.quant_output(output)

      if self.mp_group is not None:
          from deepspeed import comm as dist
          dist.inference_all_reduce(dqoutput, group=self.mp_group)
      if self.bias is not None:
          dqoutput += self.bias
      return dqoutput

  def forward_measure(self, input):
      assert input.shape[
          -1] % self.world_size == 0, 'Please ensure that self.world_size is divisible by input.shape[-1]'
      input_shard = input.shape[-1] // self.world_size
      splittedInput = input[:, :, self.rank * input_shard:(self.rank + 1) * input_shard]
      measure_input((splittedInput,), observer=self._mod_config.inputs)
      output = torch.matmul(splittedInput, self.weight.t())
      measure_output(output, self._mod_config.outputs)

      if self.mp_group is not None:
          from deepspeed import comm as dist
          dist.inference_all_reduce(output, group=self.mp_group)
      if self.bias is not None:
          output += self.bias
      return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_weight dtype={self.scale_weight.dtype}')

class PatchedKVCache(nn.Module): # TODO SW-169816 utilize original module methods to reduce code duplication
    # Module to patch KVCache module from llama model
    def __init__(self, mod, mod_config, *args, **kwargs):
        super(PatchedKVCache, self).__init__()

        # Since we don't use `self.__dict__.update`` here, need to copy the hqt config
        # TODO: [SW-178259] Add dictionary update to patched KV cache
        config = get_hqt_config(mod)
        set_hqt_config(self, config)
        self._mod_config = mod_config
        self.org_allocate = mod.allocate
        self.get_shape = mod.get_shape
        self.forward = mod.forward
        self.org_update = mod.update
        if (config.cfg['mode'] == QuantMode.QUANTIZE):
          mod.update = self.update
          self.quant_input_0 = self._mod_config.inputs[0]
          self.quant_output = self._mod_config.outputs
        elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
          mod.update = self.update_measure

    # overwrite allocate function of original module to force allocation in fp8
    def allocate(self, inp_seq_len, dtype, device, shape):
        config = get_hqt_config(self)
        dtype = torch.float8_e4m3fn if (config.cfg['mode'] == QuantMode.QUANTIZE) else dtype
        return self.org_allocate(inp_seq_len, dtype, device, shape)

    # overwrite update function of original module to force quant and dequant of cache input and output
    def update(self, prev, cur, dim, idx, inp_seq_len):
        qinput = self.quant_input_0(cur)
        output = self.org_update(prev, qinput, dim, idx, inp_seq_len)
        if output.dtype == torch.float8_e4m3fn:
          return self.quant_output(output)
        else:
           return output

    # overwrite update function of original module to force quant and dequant of cache input and output
    def update_measure(self, prev, cur, dim, idx, inp_seq_len):
        measure_input((cur,), self._mod_config.inputs)
        output = self.org_update(prev, cur, dim, idx, inp_seq_len)
        measure_output(output, self._mod_config.outputs)
        return output

class PatchedConv2d(nn.Conv2d):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      self.quant_input_0 = self._mod_config.inputs[0]
      self.scale_input = nn.Parameter(mod_config.scale.inputs[0])
      self.scale_weight = nn.Parameter(mod_config.scale.params['weight'])
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward(self, input):
    qinput = self.quant_input_0(input)
    output = conv2d_fp8(qinput, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, \
       out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
    return output

  def forward_measure(self, input):
    measure_input((input,), observer = self._mod_config.inputs)
    output = self.forward_orig(input)
    measure_output(output, self._mod_config.outputs)
    return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_weight dtype={self.scale_weight.dtype}')

class PatchedSoftmax(nn.Module):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      self.quant_output = self._mod_config.outputs
      # input scale is 1 assuming the input to SM is descaled because we are using HW supported scales
      self.scale_input = nn.Parameter(torch.Tensor([1.]))
      self.scale_output = nn.Parameter(torch.Tensor([1 / mod_config.scale.outputs]))
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward(self, x, dim = None, invAttnHead = None):
    output = torch.ops.hpu.softmax_fp8(x, dim, self.scale_input, self.scale_output, invAttnHead)
    return self.quant_output(output)

  def forward_measure(self, x, dim = None, invAttnHead = None):
    measure_input((x,), observer = self._mod_config.inputs)
    output = self.forward_orig(x, dim, invAttnHead)
    measure_output(output, self._mod_config.outputs)
    return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_output dtype={self.scale_output.dtype}')

class PatchedLoRACompatibleLinear(nn.Linear):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      self.quant_input_0 = self._mod_config.inputs[0]
      self.weight = nn.Parameter(self.weight.t().contiguous())
      self.scale_input = nn.Parameter(mod_config.scale.inputs[0])
      self.scale_weight = nn.Parameter(mod_config.scale.params['weight'])
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward(self, input, scale: float = 1.0):
    qinput = self.quant_input_0(input)
    y = matmul_fp8(qinput, self.weight, out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
    output = y+self.bias if (self.bias is not None) else y
    if self.lora_layer is not None:
      # TODO SW-174899 support lora layer quantization
      _raise_lora_layer_error(self.class_name_org)
      # output = output + (scale * self.lora_layer(input))
    return output

  def forward_measure(self, input, scale: float = 1.0):
    measure_input((input,), observer = self._mod_config.inputs)
    output = self.forward_orig(input, scale)
    measure_output(output, self._mod_config.outputs)
    return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_weight dtype={self.scale_weight.dtype}')

class PatchedLoRACompatibleConv(nn.Conv2d):
  def __init__(self, mod, mod_config, *args, **kwargs):
    self.__dict__.update(mod.__dict__)
    config = get_hqt_config(self)
    self.extra_repr_org = mod.extra_repr
    self.class_name_org = mod.__class__.__name__
    self._mod_config = mod_config
    if (config.cfg['mode'] == QuantMode.QUANTIZE):
      self.quant_input_0 = self._mod_config.inputs[0]
      self.scale_input = nn.Parameter(mod_config.scale.inputs[0])
      self.scale_weight = nn.Parameter(mod_config.scale.params['weight'])
    elif (config.cfg['mode'] == QuantMode.MEASURE) or (config.cfg['mode'] == QuantMode.SHAPE):
      self.forward=self.forward_measure

  def forward(self, input, scale: float = 1.0):
    qinput = self.quant_input_0(input)
    if self.lora_layer is not None:
      # TODO SW-174899 support lora layer quantization
      _raise_lora_layer_error(self.class_name_org)
      # output = conv2d_fp8(qinput, self.weight, None, self.stride, self.padding, self.dilation, self.groups, \
      #  out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
      # output = output + (scale * self.lora_layer(input))
      # output = output+torch.unsqueeze(torch.unsqueeze(self.bias,1), 1) if (self.bias is not None) else output
    else:
      output = conv2d_fp8(qinput, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, \
       out_dtype=self._mod_config.config_params['hp_dtype'], scale_input_inv=self.scale_input, scale_other_inv=self.scale_weight)
    return output

  def forward_measure(self, input, scale: float = 1.0):
    measure_input((input,), observer = self._mod_config.inputs)
    output = self.forward_orig(input, scale)
    measure_output(output, self._mod_config.outputs)
    return output

  def extra_repr(self) -> str:
    return extra_representation(self.extra_repr_org(), self.class_name_org, \
                                f'scale_input dtype={self.scale_input.dtype}, scale_weight dtype={self.scale_weight.dtype}')
