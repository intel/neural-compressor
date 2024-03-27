from torchvision import models

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data import DataLoader, Datasets
from neural_compressor.quantization import fit

float_model = models.resnet18()
dataset = Datasets("pytorch")["dummy"](shape=(1, 3, 224, 224))
calib_dataloader = DataLoader(framework="pytorch", dataset=dataset)
static_quant_conf = PostTrainingQuantConfig()
quantized_model = fit(model=float_model, conf=static_quant_conf, calib_dataloader=calib_dataloader)