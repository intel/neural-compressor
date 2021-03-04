import tensorflow as tf
import time
import numpy as np

tf.compat.v1.disable_eager_execution()

def main():

    import lpot
    quantizer = lpot.Quantization('./conf.yaml')

    # Get graph from slim checkpoint
    from tf_slim.nets import inception
    model_func = inception.inception_v1
    arg_scope = inception.inception_v1_arg_scope()
    kwargs = {'num_classes': 1001}
    inputs_shape = [None, 224, 224, 3]
    images = tf.compat.v1.placeholder(name='input', \
    dtype=tf.float32, shape=inputs_shape)

    
    # Do quantization
    model = quantizer.model('./inception_v1.ckpt')
    quantized_model = quantizer(model)
  
     
if __name__ == "__main__":

    main()
