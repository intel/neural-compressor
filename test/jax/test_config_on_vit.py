import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras_hub.models import ViTImageClassifier
from neural_compressor.jax.quantization.config import StaticQuantConfig

model = ViTImageClassifier.from_preset("/models/vit_base_patch16_224_imagenet")

config1 = StaticQuantConfig()
config2 = StaticQuantConfig(exclude=[".*dense_1.*", "vit_encoder/transformer_block_6/mlp/dense_2"])

model_info_1 = config1.get_model_info(model)
model_info_2 = config2.get_model_info(model)

def pretty_print(minfo1, minfo2):
    i, j = 0, 0
    def find_next_common(_i, _j):
        _ii = _i
        _jj = _j
        while _ii < len(minfo1):
            if minfo1[_ii] == minfo2[_jj]:
                return _ii, _jj
            _ii += 1
        while _jj < len(minfo2):
            if minfo1[_ii] == minfo2[_jj]:
                return _ii, _jj
            _jj += 1
        return None, None
    
    width = 80
    while i < len(minfo1) and j < len(minfo2):
        if minfo1[i] == minfo2[j]:
            print(f"{str(minfo1[i]):>{width}}   |   {str(minfo2[j]):>{width}}")
            i += 1
            j += 1
        else:
            next_i, next_j = find_next_common(i, j)
            if next_i is not None and next_j is not None:
                for k in range(i, next_i):
                    print(f"{str(minfo1[k]):>{width}}   |")
                for k in range(j, next_j):
                    print(f"{'':>{width}}   |   {str(minfo2[k]):>{width}}")
                i = next_i
                j = next_j
            else:
                print(f"{str(minfo1[i]):>{width}}   |   {str(minfo2[j]):>{width}}")
                i += 1
                j += 1

pretty_print(model_info_1, model_info_2)