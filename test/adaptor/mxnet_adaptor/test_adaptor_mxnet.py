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
from tempfile import TemporaryDirectory

from neural_compressor.experimental import Quantization, common
from neural_compressor.experimental.metric.metric import MXNetMetrics
from neural_compressor.utils.utility import recover
from neural_compressor.adaptor.mxnet_utils.util import check_mx_version

WORKSPACE_DIR = Path('./saved')

MX_NAMESPACE = mx.np if check_mx_version('2.0.0') else mx.nd


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

  nodes_a = [(node['op'], node['inputs']) for node in json.loads(symnet_a.tojson())['nodes']]
  nodes_b = [(node['op'], node['inputs']) for node in json.loads(symnet_b.tojson())['nodes']]
  tester.assertEqual(nodes_a, nodes_b)

  args_a = dict(sorted(args_a.items(), key=lambda x: x[0]))
  args_b = dict(sorted(args_b.items(), key=lambda x: x[0]))
  auxs_a = dict(sorted(auxs_a.items(), key=lambda x: x[0]))
  auxs_b = dict(sorted(auxs_b.items(), key=lambda x: x[0]))

  assert len(args_a) == len(args_b)
  for val_a, val_b in zip(args_a.values(), args_b.values()):
    tester.assertTrue(np.all((val_a == val_b).asnumpy()))

  assert len(auxs_a) == len(auxs_b)
  for val_a, val_b in zip(auxs_a.values(), auxs_b.values()):
    tester.assertTrue(np.all((val_a == val_b).asnumpy()))


class TestAdaptorMXNet(unittest.TestCase):
    """
    Test MXNet adaptor functions.
    """
    @classmethod
    def setUpClass(self):
      build_mxnet()
      build_mxnet_kl()

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
      mlp_input = mx.symbol.Variable('data')
      mlp_model = mx.symbol.FullyConnected(data=mlp_input, name='fc1', num_hidden=32)
      mlp_model = mx.symbol.Activation(data=mlp_model, act_type='relu')
      mlp_model = mx.symbol.FullyConnected(data=mlp_model, name='fc2', num_hidden=16)
      mlp_model = mx.symbol.Softmax(mlp_model, name='softmax')

      for shape in [(32, 64), ]:
        data = MX_NAMESPACE.random.uniform(
            self.data_low, self.data_high, shape).astype('float32')
        labels = MX_NAMESPACE.ones((shape[0],))
        calib_data = mx.io.NDArrayIter(data=data, label=labels, batch_size=shape[0])

        with TemporaryDirectory() as tmpdirname:
          prefix = str(Path(tmpdirname)/'tmp')
          sym_block = mx.gluon.SymbolBlock(mlp_model, [mlp_input])
          sym_block.initialize()
          sym_block.forward(data)
          sym_block.export(prefix, epoch=0)
          fp32_model = mx.model.load_checkpoint(prefix, 0)

        quantizer = Quantization("./mxnet.yaml")
        quantizer.model = fp32_model
        quantizer.calib_dataloader = calib_data
        quantizer.eval_dataloader = calib_data
        qmodel = quantizer.fit()
        self.assertIsInstance(qmodel.model[0], mx.symbol.Symbol)

        # test inspect_tensor
        inspect_tensor = quantizer.strategy.adaptor.inspect_tensor
        quantizer.model = fp32_model

        fc_op_name = 'sg_{}_fully_connected'.format(
            'onednn' if check_mx_version('2.0.0') else 'mkldnn')
        fc_node_name1 = fc_op_name + '_eltwise_0'
        fc_node_name2 = fc_op_name + '_1'

        insp = inspect_tensor(quantizer.model, quantizer.calib_dataloader,
                              op_list=[(fc_node_name1, fc_op_name),
                                       (fc_node_name2, fc_op_name)], iteration_list=[1, 3])
        qinsp = inspect_tensor(qmodel, quantizer.calib_dataloader,
                               op_list=[(fc_node_name1, fc_op_name),
                                        (fc_node_name2, fc_op_name)], iteration_list=[1, 3])

        self.assertNotEqual(len(insp['activation']), 0)
        self.assertEqual(len(insp['activation']), len(qinsp['activation']))

        for tensors, qtensors in zip(insp['activation'], qinsp['activation']):
          for k in (set(tensors.keys()) & set(qtensors.keys())):
            tensor, qtensor = tensors[k][k[0]], qtensors[k][k[0]]
            self.assertEqual(tensor.shape, qtensor.shape)

        #test inspect with an empty iteration_list
        inspect_tensor(qmodel, quantizer.calib_dataloader,
                       op_list=[(fc_node_name1, fc_op_name)],
                       iteration_list=[])

        # test recovery for symbolic model
        qmodel_r = recover(fp32_model, WORKSPACE_DIR/'history.snapshot', -1)
        are_models_equal(self, qmodel.model, qmodel_r.model)

        # test symbolic model saving
        qmodel_r.save(WORKSPACE_DIR/'save_test')

    def test_conv_model_quantization(self):
      """
      Use Conv model to test KL calibration and user specific evaluate function.
      """
      conv_net = nn.HybridSequential()
      conv_net.add(nn.Conv2D(channels=3, kernel_size=(1, 1)))
      conv_net.add(nn.BatchNorm())
      conv_net.add(nn.Activation('relu'))
      conv_net.add(nn.AvgPool2D(pool_size=(4, 4)))
      conv_net.add(nn.Dense(1, activation='sigmoid'))
      conv_net.initialize()

      for shape in [(32, 3, 224, 224), ]:
        dataShape = (shape[0]*5, *shape[1:])
        data = MX_NAMESPACE.random.uniform(self.data_low, self.data_high, dataShape,
                                           dtype='float32')
        label = MX_NAMESPACE.random.randint(0, 2, (dataShape[0], 1)).astype('float32')
        dataset = mx.gluon.data.ArrayDataset(data, label)

        def eval(model):
          eval_dataloader = mx.gluon.data.DataLoader(dataset, batch_size=8)
          metric = MXNetMetrics().metrics['Accuracy']()
          for batch in eval_dataloader:
            data, labels = batch
            preds = model.forward(data)
            metric.update(labels.asnumpy(), preds.asnumpy())
          return metric.result()

        calib_dataloader = mx.gluon.data.DataLoader(dataset, batch_size=8)
        calib_dataloader.batch_size = 8
        quantizer = Quantization("./mxnet_kl.yaml")
        quantizer.model = conv_net
        quantizer.calib_dataloader = calib_dataloader
        quantizer.eval_func = eval
        qnet = quantizer.fit().model
        self.assertIsInstance(qnet, mx.gluon.HybridBlock)

    def test_gluon_model(self):
      """
      Use gluon model to test gluon related functions in mxnet adaptor.
      """
      # create gluon model
      def create_model(params=None):
          net = nn.HybridSequential()
          net.add(nn.Conv2D(1, (1, 1), activation="relu"))
          net.add(nn.Flatten())
          net.add(nn.Dense(64, activation="relu"))
          net.add(nn.Dense(10))
          if params is not None:
            if check_mx_version('2.0.0'):
              net.load_dict({k: v.data() for k, v in params.items()})
            else:
              param_keys = sorted(net.collect_params().keys())
              param_values = sorted(params.items(), key=lambda x: x[0])
              params = {k: v.data() for k, (old_k, v) in zip(param_keys, param_values)}
              net.collect_params().load_dict(params)
          else:
            net.initialize()
          return net

      class CalibDataset():
        def __init__(self, dataset):
          self.dataset = dataset

        def __getitem__(self, idx):
          if check_mx_version('2.0.0'):
            mx_namespace = mx.np
          else:
            mx_namespace = mx.nd
          data, label = self.dataset[idx]
          data = mx_namespace.reshape(
              data, (data.shape[-1], *data.shape[:-1])).astype('float32')
          return data, label

        def __len__(self):
          return len(self.dataset)

      net = create_model()
      dataset = CalibDataset(mx.gluon.data.vision.datasets.FashionMNIST(train=False))
      dataloader = common.DataLoader(dataset, batch_size=8)
      quantizer = Quantization("./mxnet.yaml")
      quantizer.model = net
      quantizer.calib_dataloader = dataloader
      quantizer.eval_dataloader = dataloader
      qnet = quantizer.fit()
      self.assertIsInstance(qnet.model, mx.gluon.HybridBlock)

      # test recovery for gluon model
      net = create_model(net.collect_params())
      qnet_r = recover(net, WORKSPACE_DIR/'history.snapshot', -1)

      from neural_compressor.adaptor.mxnet_utils.util import prepare_model, prepare_dataloader
      dataloader = prepare_dataloader(qnet, mx.cpu(), quantizer.calib_dataloader)

      # test calling prepare_dataloader for already prepared dataloader
      self.assertIs(dataloader, prepare_dataloader(qnet, mx.cpu(), dataloader))

      model_a = prepare_model(qnet, mx.cpu(), dataloader.input_desc)
      model_b = prepare_model(qnet_r, mx.cpu(), dataloader.input_desc)
      are_models_equal(self, model_a, model_b)

      # test gluon model saving
      qnet_r.save(WORKSPACE_DIR/'save_test')


if __name__ == "__main__":
    unittest.main()
