import os
quant_config=os.getenv('QUANT_CONFIG', None) is not None
print(f"QUANT_CONFIG enabled: {quant_config}")
if quant_config:
  from .prepare_quant.prepare_model import prep_model, finish_measurements
  from .prepare_quant.prepare_model import finish_measurements
  from .run_quant.quantization import quantize_model
else:
  def prep_model(*args, **kwargs):
    return
  def finish_measurements(*args, **kwargs):
    return
  def quantize_model(*args, **kwargs):
    return
