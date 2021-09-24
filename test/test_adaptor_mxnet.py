import unittest
import os
import sys
import shutil
import yaml
import json
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn
from pathlib import Path

from neural_compressor.experimental import Quantization, common
from neural_compressor.utils.utility import recover


sys.path.append('..')

WORKSPACE_DIR = Path('./saved')


def get_mlp_sym():
    data = mx.symbol.Variable('data')
    out = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=1000)
    out = mx.symbol.Activation(data=out, act_type='relu')
    out = mx.symbol.Softmax(out, name='softmax')
    return out


def get_conv_sym():
    data = mx.sym.Variable('data')
    conv = mx.sym.Convolution(data, kernel=(1, 1), num_filter=3, name='conv')
    batch_norm = mx.sym.BatchNorm(data=conv, eps=2e-05, fix_gamma=False,
                                  momentum=0.9, use_global_stats=False, name='bn')
    act = mx.sym.Activation(data=batch_norm, act_type='relu', name='relu')
    pool = mx.sym.Pooling(act, kernel=(4, 4), pool_type='avg', name='pool')
    out = mx.sym.FullyConnected(pool, num_hidden=10, flatten=True, name='fc')

    return out


def build_mxnet():
    fake_yaml = '''
        model:
          name: imagenet
          framework: mxnet

        evaluation:
          accuracy:
            metric:
              topk: 1

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: {}
        '''.format(str(WORKSPACE_DIR))
    configs = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('mxnet.yaml', "w", encoding="utf-8") as f:
        yaml.dump(configs, f)
    f.close()


def build_mxnet_kl():
    fake_yaml = '''
        model:
          name: imagenet
          framework: mxnet

        evaluation:
          accuracy:
            metric:
              topk: 1

        quantization:
          model_wise:
            activation:
              algorithm: kl

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: {}
        '''.format(str(WORKSPACE_DIR))
    configs = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('mxnet_kl.yaml', "w", encoding="utf-8") as f:
        yaml.dump(configs, f)
    f.close()


def are_models_equal(tester, model_a, model_b):
  symnet_a, args_a, auxs_a = model_a
  symnet_b, args_b, auxs_b = model_b

  nodes_a = {node['name']: node['op'] for node in json.loads(model_a[0].tojson())['nodes']}
  nodes_b = {node['name']: node['op'] for node in json.loads(model_b[0].tojson())['nodes']}
  tester.assertEqual(nodes_a, nodes_b)

  tester.assertEqual(args_a.keys(), args_b.keys())
  for key in args_a.keys():
    tester.assertTrue(np.all((args_a[key] == args_b[key]).asnumpy()))

  tester.assertEqual(auxs_a.keys(), auxs_b.keys())
  for key in auxs_a.keys():
    tester.assertTrue(np.all((auxs_a[key] == auxs_b[key]).asnumpy()))


class TestAdaptorMXNet(unittest.TestCase):
    """
    Test MXNet adaptor functions.
    """
    @classmethod
    def setUpClass(self):
        build_mxnet()
        build_mxnet_kl()
        self.mlp_model = get_mlp_sym()
        self.conv_model = get_conv_sym()
        self.quantizer_1 = Quantization("./mxnet.yaml")
        self.quantizer_2 = Quantization("./mxnet_kl.yaml")

        self.data_low = -1000
        self.data_high = 1000

    @classmethod
    def tearDownClass(self):
        os.remove('mxnet.yaml')
        os.remove('mxnet_kl.yaml')
        shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_utils(self):
      import neural_compressor.adaptor.mxnet_utils.util as utils
      self.assertTrue(utils.isiterable([1, 2, 3]))
      self.assertFalse(utils.isiterable(123))



    def test_mlp_model_quantization(self):
        """
        Use MLP model to test minmax calibration and built-in evaluate function.
        """
        for shape in [(500, 1000), ]:
            arg_shapes, label_shape, _ = self.mlp_model.infer_shape(data=shape)

            mod = mx.mod.Module(symbol=self.mlp_model, context=mx.current_context())
            mod.bind(for_training=False,
                     data_shapes=[('data', arg_shapes[0])],
                     label_shapes=[('softmax_label', label_shape[0])])
            mod.init_params()

            arg_params, aux_params = mod.get_params()
            data = mx.nd.random.uniform(low=self.data_low, high=self.data_high,
                                        shape=shape).astype('float32')
            labels = mx.nd.ones([shape[0], ])
            calib_data = mx.io.NDArrayIter(data=data, label=labels, batch_size=shape[0])

            fp32_model = (self.mlp_model, arg_params, aux_params)
            self.quantizer_1.model = common.Model(fp32_model)
            self.quantizer_1.calib_dataloader = calib_data
            self.quantizer_1.eval_dataloader = calib_data
            qmodel = self.quantizer_1()
            self.assertIsInstance(qmodel.model[0], mx.symbol.Symbol)

    def test_conv_model_quantization(self):
        """
        Use Conv model to test KL calibration and user specific evaluate function.
        """
        for shape in [(32, 3, 224, 224), ]:
            arg_shapes, _, _ = self.conv_model.infer_shape(data=shape)

            mod = mx.mod.Module(symbol=self.conv_model, context=mx.current_context())
            mod.bind(for_training=False,
                     data_shapes=[('data', arg_shapes[0])])
            mod.init_params()

            arg_params, aux_params = mod.get_params()
            dataShape = (shape[0]*5, *shape[1:])
            data = mx.nd.random.uniform(low=self.data_low, high=self.data_high,
                                        shape=dataShape, dtype='float32')
            label = mx.nd.random.randint(low=0, high=1, shape=(dataShape[0])).astype('float32')
            calib_data = mx.io.NDArrayIter(data=data, label=label,  batch_size=shape[0])

            fp32_model = (self.conv_model, arg_params, aux_params)
            self.quantizer_2.model = common.Model(fp32_model)
            self.quantizer_2.calib_dataloader = calib_data
            self.quantizer_2.eval_dataloader = calib_data
            qmodel = self.quantizer_2()
            self.assertIsInstance(qmodel.model[0], mx.symbol.Symbol)

            # test inspected_tensor
            inspect_tensor = self.quantizer_2.strategy.adaptor.inspect_tensor
            self.quantizer_2.model = fp32_model
            insp = inspect_tensor(self.quantizer_2.model, calib_data,
                                  op_list=[('sg_mkldnn_conv_bn_act_0', 'sg_mkldnn_conv'),
                                           ('data', 'input')],
                                  iteration_list=[1, 3])
            qinsp = inspect_tensor(qmodel, calib_data,
                                   op_list=[('sg_mkldnn_conv_bn_act_0', 'sg_mkldnn_conv')],
                                   iteration_list=[1, 3])

            self.assertNotEqual(len(insp['activation']), 0)
            self.assertEqual(len(insp['activation']), len(qinsp['activation']))

            for tensors, qtensors in zip(insp['activation'], qinsp['activation']):
              for k in (set(tensors.keys()) & set(qtensors.keys())):
                tensor, qtensor = tensors[k][k[0]], qtensors[k][k[0]]
                self.assertEqual(tensor.shape, qtensor.shape)

            #test inspect with an empty iteration_list
            inspect_tensor(qmodel, calib_data,
                           op_list=[('sg_mkldnn_conv_bn_act_0', 'sg_mkldnn_conv')],
                           iteration_list=[])

            # test recovery for symbolic model
            qmodel_r = recover(fp32_model, WORKSPACE_DIR/'history.snapshot', -1)
            are_models_equal(self, qmodel.model, qmodel_r.model)

            # test symbolic model saving
            qmodel_r.save(WORKSPACE_DIR/'save_test')

    def test_gluon_model(self):
        """
        Use gluon model to test gluon related functions in mxnet adaptor.
        """
        # create gluon model
        def create_model(params=None):
          net = nn.HybridSequential()
          net.add(nn.Conv2D(1, (1, 1), activation="relu", prefix='conv1'))
          net.add(nn.Flatten(prefix='flatten1'))
          net.add(nn.Dense(64, activation="relu", prefix='dense1'))
          net.add(nn.Dense(10, prefix='dense2'))
          if params is not None:
            net.collect_params().load_dict({k: v.data() for k, v in params.items()})
          else:
            net.initialize()
          return net

        class Quant_dataloader():
            def __init__(self, dataset, batch_size=1):
                self.dataset = dataset
                self.batch_size = batch_size
                self._iter = None

            def __iter__(self):
              self._iter = iter(self.dataset)
              return self

            def __next__(self):
              data, label = self._iter.__next__()
              data = mx.nd.reshape(data, (1, data.shape[-1], *data.shape[:-1])).astype('float32')
              return (data, mx.nd.array(label, dtype='float32'))

        net = create_model()
        valid_dataset = mx.gluon.data.vision.datasets.FashionMNIST(train=False)
        q_dataloader = Quant_dataloader(valid_dataset)
        self.quantizer_1.model = common.Model(net)
        self.quantizer_1.calib_dataloader = q_dataloader
        qmodel = self.quantizer_1()
        self.assertIsInstance(qmodel.model, mx.gluon.HybridBlock)

        # test recovery for gluon model
        net = create_model(net.collect_params())
        qmodel_r = recover(net, WORKSPACE_DIR/'history.snapshot', -1)

        from neural_compressor.adaptor.mxnet_utils.util import prepare_model, prepare_dataloader
        dataloader = prepare_dataloader(qmodel, mx.cpu(), q_dataloader)

        # test calling prepare_dataloader for already prepared dataloader
        self.assertIs(dataloader, prepare_dataloader(qmodel, mx.cpu(), dataloader))

        model_a = prepare_model(qmodel, mx.cpu(), dataloader.input_desc)
        model_b = prepare_model(qmodel_r, mx.cpu(), dataloader.input_desc)
        are_models_equal(self, model_a, model_b)

        # test gluon model saving
        qmodel_r.save(WORKSPACE_DIR/'save_test')


if __name__ == "__main__":
    unittest.main()
