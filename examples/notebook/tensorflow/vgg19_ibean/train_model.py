import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import neural_compressor as inc
print("neural_compressor version {}".format(inc.__version__))

import tensorflow as tf
print("tensorflow {}".format(tf.__version__))

import matplotlib.pyplot as plt
import numpy as np

# define class number
class_num=3

# define input image size and class number
w=h=32

def load_raw_dataset():
    raw_datasets, raw_info = tfds.load(name = 'beans', with_info = True,
                                       as_supervised = True, 
                                       split = ['train', 'test'])
    return raw_datasets, raw_info

def preprocess(image, label):
    image = tf.cast(image, tf.float32)/255.0
    return tf.image.resize(image, [w, h]), tf.one_hot(label, class_num)

def load_dataset(batch_size = 32):
    datasets, info = load_raw_dataset()
    return [dataset.map(preprocess).batch(batch_size) for dataset in datasets]

def build_model(w, h, class_num):
    url = 'https://tfhub.dev/deepmind/ganeval-cifar10-convnet/1'
    feature_extractor_layer = hub.KerasLayer(url, input_shape = (w, h, 3))
    feature_extractor_layer.trainable = False

    model = tf.keras.Sequential(
        [
            feature_extractor_layer,
            #tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(class_num, activation = 'softmax')
        ]
    )

    model.summary()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
        metrics = ['acc']
    )    
    return model

def train_model(model, epochs=1):
    train_dataset, test_dataset = load_dataset()
    hist = model.fit(train_dataset, epochs = epochs, validation_data = test_dataset)
    result = model.evaluate(test_dataset)
    
def save_model(model, model_path):    
    model.save(model_path)
    print("Save model to {}".format(model_path))
    


model = build_model(w, h, class_num)
epochs=2
train_model(model, epochs)
model_fp32_path="model_keras.fp32"
save_model(model, model_fp32_path)


