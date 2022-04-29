import tensorflow as tf
from neural_compressor.experimental import Quantization, common

tf.compat.v1.disable_eager_execution()

def main():

    quantizer = Quantization()
    quantizer.model = './mobilenet_v1_1.0_224_frozen.pb'
    dataset = quantizer.dataset('dummy', shape=(20, 224, 224, 3))
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantized_model = quantizer.fit()


if __name__ == "__main__":

    main()