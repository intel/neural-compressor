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

    from lpot.adaptor.tf_utils.util import get_slim_graph
    graph = get_slim_graph('./inception_v1.ckpt', model_func, \
            arg_scope, images, **kwargs)
    
    # Do quantization
    quantized_model = quantizer(graph)
  
     
if __name__ == "__main__":

    main()
