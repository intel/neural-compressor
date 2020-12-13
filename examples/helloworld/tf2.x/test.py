import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np


def get_concrete_function(graph_def, inputs, outputs, print_graph=False):
    def imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrap_function = tf.compat.v1.wrap_function(imports_graph_def, [])
    graph = wrap_function.graph

    return wrap_function.prune(
        tf.nest.map_structure(graph.as_graph_element, inputs),
        tf.nest.map_structure(graph.as_graph_element, outputs))


def main():

    # Get data
    (train_images, train_labels), (test_images,
                                    test_labels) = keras.datasets.fashion_mnist.load_data()

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Load saved model
    model = tf.keras.models.load_model("../models/simple_model")

    print('input', model.input_names)
    print('output', model.output_names)

    # Run lpot to get the quantized graph 
    import lpot
    quantizer = lpot.Quantization('./conf.yaml')
    dataloader = quantizer.dataloader(dataset=list(zip(test_images, test_labels)))
    quantized_model = quantizer(model, q_dataloader=dataloader, eval_dataloader=dataloader)

    # Run inference with quantized model
    concrete_function = get_concrete_function(graph_def=quantized_model.as_graph_def(),
                                     inputs=["input:0"],
                                     outputs=["output:0"],
                                     print_graph=True)

    frozen_graph_predictions = concrete_function(input=tf.constant(test_images))[0]
    print("Inference is done.")
    
if __name__ == "__main__":

    main()
