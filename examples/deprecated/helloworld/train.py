import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def main():

    # Get data
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Create Keras model
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28), name="input"),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation="softmax", name="output")
    ])

    # Print model architecture
    model.summary()

    # Compile model with optimizer
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

    # Train model
    model.fit(x={"input": train_images}, y={"output": train_labels}, epochs=1)
    model.save("./models/saved_model")

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    # Generate frozen pb
    tf.io.write_graph(graph_or_graph_def=frozen_model.graph,
                       logdir="./models",
                       name="frozen_graph.pb",
                       as_text=False)

    return

if __name__ == "__main__":
    main()
