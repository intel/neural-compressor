import tensorflow as tf
import time
from tensorflow import keras
import numpy as np
from ilit import Quantization

def eval_func(model):
    return 1.

def load_graph(model_file):
  """This is a function to load TF graph from pb file

  Args:
      model_file (string): TF pb file local path

  Returns:
      graph: TF graph object
  """
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
     graph_def.ParseFromString(f.read())

  with graph.as_default():
    tf.import_graph_def(graph_def, name='')

  return graph

def main():
    # Get data
    (train_images, train_labels), (test_images,
                                    test_labels) = keras.datasets.fashion_mnist.load_data()

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Load model
    model_file = "../frozen_models/simple_frozen_graph.pb"
    graph = load_graph(model_file)

    # Run ilit to get the quantized pb
    quantizer = Quantization('./conf.yaml')
    dataloader = quantizer.dataloader(dataset=(test_images, test_labels))
    quantized_model = quantizer(graph, q_dataloader=dataloader, eval_func=eval_func)

    # Run quantized model 
    with tf.compat.v1.Session() as sess:
        tf.compat.v1.import_graph_def(quantized_model.as_graph_def())
        styled_image = sess.run(['import/Identity:0'], feed_dict={'import/x:0':test_images})
        print("Inference is done.")

if __name__ == "__main__":

    main()
