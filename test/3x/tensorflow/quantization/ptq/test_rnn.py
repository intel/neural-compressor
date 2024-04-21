import os
import unittest

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.compat.v1 import graph_util

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphRewriterHelper as Helper
from neural_compressor.tensorflow.utils import disable_random


def quantize(model, q_data, e_data):
    from neural_compressor.tensorflow import quantize_model
    from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset

    calib_dataloader = BaseDataLoader(dataset=list(zip(q_data[0], q_data[1])))
    quant_config = {
        "static_quant": {
            "global": {
                "weight_granularity": "per_tensor",
                "act_algorithm": "minmax",
            },
        }
    }
    q_model = quantize_model(model, quant_config, calib_dataloader)

    return q_model


class TestTensorflowRnn(unittest.TestCase):
    @unittest.skipUnless(bool(tf.version.VERSION.find("1.15.0-up2") != -1), "not supported the current tf version.")
    @disable_random()
    def test_tensorflow_dynamic_rnn(self):
        X = np.random.randn(3, 6, 4)

        X[1, 4:] = 0
        X_lengths = [6, 4, 6]

        rnn_hidden_size = 5
        rnn_type = "ltsm1"
        if rnn_type == "lstm":
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.GRUCell(num_units=rnn_hidden_size)

        outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float64, sequence_length=X_lengths, inputs=X)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            o1, s1 = sess.run([outputs, last_states])
            rs = Helper.analysis_rnn_model(sess.graph.as_graph_def())
            self.assertEqual(len(rs.keys()), 2)

    @unittest.skipUnless(bool(tf.version.VERSION.find("1.15.0-up2") != -1), "not supported the current tf version.")
    @disable_random()
    def test_tensorflow_rnn(self):
        inp = tf.keras.layers.Input(shape=(None, 4))
        lstm_1 = tf.keras.layers.LSTM(units=10, return_sequences=True)(inp)
        dropout_1 = tf.keras.layers.Dropout(0.2)(lstm_1)
        lstm_2 = tf.keras.layers.LSTM(units=10, return_sequences=False)(dropout_1)
        dropout_2 = tf.keras.layers.Dropout(0.2)(lstm_2)
        out = tf.keras.layers.Dense(1)(dropout_2)
        model = tf.keras.models.Model(inputs=inp, outputs=out)

        model.compile(loss="mse", optimizer=tf.keras.optimizers.RMSprop())

        input_names = [t.name.split(":")[0] for t in model.inputs]
        output_names = [t.name.split(":")[0] for t in model.outputs]

        q_data = np.random.randn(64, 10, 4)
        label = np.random.randn(64, 1)
        model.predict(q_data)

        sess = tf.keras.backend.get_session()

        graph = sess.graph
        graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            output_names,
        )
        with tf.Graph().as_default() as g:
            tf.import_graph_def(graph_def, name="")
            s = quantize(g, q_data=(q_data, label), e_data=(q_data, label))

        convert_count = 0
        for i in s.graph_def.node:
            if i.op == "QuantizedMatMulWithBiasAndDequantize":
                convert_count += 1
        self.assertEqual(convert_count, 9)


if __name__ == "__main__":
    unittest.main()
