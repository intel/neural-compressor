# First, run measurements (based on custom_config/measure_config.json)
QUANT_CONFIG=measure_config python3 imagenet_quant.py

# Next, run the quantized model (based on custom_config/quant_config.json)
QUANT_CONFIG=quant_config python3 imagenet_quant.py