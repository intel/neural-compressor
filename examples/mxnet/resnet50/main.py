from mxnet_model-zoo import rn50
import LPiT

model = rn50()
at = LPiT.AutoTuner("user.yaml")
best_quantized_model = at.tune(model)



