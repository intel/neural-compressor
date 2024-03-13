from transformers import AutoModel

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit

float_model = AutoModel.from_pretrained("bert-base-uncased")
woq_conf = PostTrainingQuantConfig(approach="weight_only")
quantized_model = fit(model=float_model, conf=woq_conf)
