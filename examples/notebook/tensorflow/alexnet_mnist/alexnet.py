import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow import keras

import mnist_dataset


def save_mod(model, mod_path):
    print('Save to {}'.format(mod_path))
    tf.saved_model.save(model, mod_path)


def load_mod(model_file):
    model = tf.keras.models.load_model(model_file)
    print('Load from {}'.format(model_file))
    return model

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


def load_pb(in_model):
    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(in_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(od_graph_def, name='')

    return detection_graph

def read_data():
    x_train, y_train, label_train, x_test, y_test, label_test = mnist_dataset.read_data()
    return x_train, y_train, label_train, x_test, y_test, label_test

def create_model(w, c, classes):
    model = Sequential()
    model.add(Convolution2D(96, 11, input_shape=(w, w, c), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(384, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(384, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 3, padding='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 7))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_mod(model, data, epochs=3):
    x_train, y_train, label_train, x_test, y_test, label_test = data
    model.fit(x_train, y_train, epochs=epochs, batch_size=600, validation_data=(x_test, y_test), verbose=1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def main():
    data = read_data()

    classes = 10
    w = 28
    c = 1
    model = create_model(w ,c, classes)
    model.summary()

    epochs = 3
    train_mod(model, data, epochs)
    save_mod(model, "alexnet_mnist_fp32_mod")

if __name__ == "__main__":
    main()
