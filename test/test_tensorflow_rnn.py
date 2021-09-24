
import unittest
import os
import numpy as np
import yaml
import tensorflow as tf
from tensorflow.python.framework import graph_util
from neural_compressor.adaptor.tf_utils.util import disable_random


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input_1
          outputs: dense/BiasAdd
        device: cpu
        quantization:
          model_wise:
            weight:
                granularity: per_tensor
            activation:
                algorithm: minmax
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: basic
            accuracy_criterion:
              relative: 0.01
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

def quantize(model,q_data, e_data):
    from neural_compressor.experimental import Quantization , common
    from neural_compressor.experimental.common import DataLoader

    quantizer = Quantization('fake_yaml.yaml')

    q_dataloader = DataLoader(dataset=list(zip(q_data[0], q_data[1])))
    e_dataloader = DataLoader(dataset=list(zip(e_data[0], e_data[1])))
    quantizer.model= common.Model(model)
    quantizer.calib_dataloader = q_dataloader
    quantizer.eval_dataloader = e_dataloader
    quantized_model = quantizer()
    return quantized_model

class TestTensorflowRnn(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')

    @unittest.skipUnless(bool(
            tf.version.VERSION.find('1.15.0-up2') != -1), 'not supported the current tf version.')
    @disable_random()
    def test_tensorflow_rnn(self):
        inp = tf.keras.layers.Input(shape=(None, 4))
        lstm_1 = tf.keras.layers.LSTM(units=10,
                    return_sequences=True)(inp)
        dropout_1 = tf.keras.layers.Dropout(0.2)(lstm_1)
        lstm_2 = tf.keras.layers.LSTM(units=10,
                    return_sequences=False)(dropout_1)
        dropout_2 = tf.keras.layers.Dropout(0.2)(lstm_2)
        out = tf.keras.layers.Dense(1)(dropout_2)
        model = tf.keras.models.Model(inputs=inp, outputs=out)

        model.compile(loss="mse",
                    optimizer=tf.keras.optimizers.RMSprop())

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
            tf.import_graph_def(graph_def, name='')
            s = quantize(g,
                         q_data=(q_data, label),
                         e_data=(q_data, label))

        convert_count = 0
        for i in s.graph_def.node:
            if i.op == 'QuantizedMatMulWithBiasAndDequantize':
                convert_count += 1
        self.assertEqual(convert_count, 9)


if __name__ == "__main__":
    unittest.main()
