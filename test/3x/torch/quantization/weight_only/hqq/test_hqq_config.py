from neural_compressor.torch.algorithms.weight_only.hqq.config import default_hqq_module_config
from neural_compressor.torch.algorithms.weight_only.hqq.qtensor import QTensorMetaInfo

out_feats = 5
in_feats = 4
meta_config = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
print(meta_config)
print(meta_config.to_dict())
print(default_hqq_module_config)
