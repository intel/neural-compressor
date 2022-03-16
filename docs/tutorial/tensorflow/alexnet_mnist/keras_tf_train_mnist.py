import os
import tensorflow as tf
print("Tensorflow version {}".format(tf.__version__))
tf.compat.v1.enable_eager_execution()

"""
Intel Neural Compressor has old package names: iLiT and LPOT.
"""
try:
    import neural_compressor as inc
    print("neural_compressor version {}".format(inc.__version__))  
except:
    try:
        import lpot as inc
        print("LPOT version {}".format(inc.__version__)) 
    except:
        import ilit as inc
        print("iLiT version {}".format(inc.__version__))       

import matplotlib.pyplot as plt
import numpy as np

"""
Dataset
Use MNIST dataset to recognize hand writing numbers. Load the dataset.
"""
import alexnet
 
data = alexnet.read_data()
x_train, y_train, label_train, x_test, y_test, label_test = data
print('train', x_train.shape, y_train.shape, label_train.shape)
print('test', x_test.shape, y_test.shape, label_test.shape)

"""
Build Model
Build a CNN model like Alexnet by Keras API based on Tensorflow. Print the model structure by Keras API: summary().
"""

classes = 10
width = 28
channels = 1

model = alexnet.create_model(width ,channels ,classes)

model.summary()

"""
Train the Model with the Dataset
Set the epochs to "1"
"""
epochs = 1

alexnet.train_mod(model, data, epochs)

"""
Freeze and Save Model to Single PB
Set the input node name is "x".
"""
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def save_frozen_pb(model, mod_path):
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    # Generate frozen pb
    tf.io.write_graph(graph_or_graph_def=frozen_model.graph,
                      logdir=".",
                      name=mod_path,
                      as_text=False)
fp32_frozen_pb_file = "fp32_frozen.pb"
save_frozen_pb(model, fp32_frozen_pb_file)

os.system("ls -la fp32_frozen.pb")
         

