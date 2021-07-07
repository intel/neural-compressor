import unittest
import os
import sys
import shutil
import yaml
import mxnet as mx
import mxnet.gluon.nn as nn

sys.path.append('..')
import lpot
from lpot.experimental import Quantization, common

def get_mlp_sym():
    data = mx.symbol.Variable('data')
    out = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=1000)
    out = mx.symbol.Activation(data=out, act_type='relu')
    out = mx.symbol.Softmax(out, name='softmax')
    return out

def get_conv_sym():
    data = mx.sym.Variable('data')
    conv = mx.sym.Convolution(data, kernel=(1, 1), num_filter=3, name='conv')
    batch_norm = mx.sym.BatchNorm(data=conv, eps=2e-05, fix_gamma=False, \
                                  momentum=0.9, use_global_stats=False, name='bn')
    act = mx.sym.Activation(data=batch_norm, act_type='relu', name='relu')
    pool = mx.sym.Pooling(act, kernel=(4, 4), pool_type='avg', name='pool')
    out = mx.sym.FullyConnected(pool, num_hidden=10, flatten=True, name='fc')

    return out

def eval_func(model):
    return 1.0

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
            path: ./saved
        '''
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
            weight:
              dtype: int8
              scheme: asym
              algorithm: minmax
            activation:
              dtype: int8
              scheme: asym
              algorithm: kl

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
          workspace:
            path: ./saved
        '''
    configs = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('mxnet_kl.yaml', "w", encoding="utf-8") as f:
        yaml.dump(configs, f)
    f.close()

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
        framework_specific_info = dict()
        framework_specific_info['q_dataloader'] = None
        self.adaptor = lpot.adaptor.mxnet.MxNetAdaptor(framework_specific_info)
        self.data_low = -1000
        self.data_high = 1000

    @classmethod
    def tearDownClass(self):
        os.remove('mxnet.yaml')
        os.remove('mxnet_kl.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_mlp_model_quantization(self):
        """
        Use MLP model to test minmax calibration and built-in evaluate function.
        """
        for shape in [(500, 1000),]:
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
        for shape in [(500, 3, 224, 224),]:
            arg_shapes, _, _ = self.conv_model.infer_shape(data=shape)

            mod = mx.mod.Module(symbol=self.conv_model, context=mx.current_context())
            mod.bind(for_training=False,
                     data_shapes=[('data', arg_shapes[0])])
            mod.init_params()

            arg_params, aux_params = mod.get_params()
            data = mx.nd.random.uniform(low=self.data_low, high=self.data_high,
                                        shape=shape).astype('float32')
            calib_data = mx.io.NDArrayIter(data=data, batch_size=shape[0])

            fp32_model = (self.conv_model, arg_params, aux_params)
            self.quantizer_2.model = common.Model(fp32_model)
            self.quantizer_2.calib_dataloader = calib_data
            self.quantizer_2.eval_dataloader = calib_data
            self.quantizer_2.eval_func = eval_func
            qmodel = self.quantizer_2()
            # test inspected_tensor
            inspect_tensor = self.quantizer_2.strategy.adaptor.inspect_tensor
            self.quantizer_2.model = fp32_model
            inspected_tensor = inspect_tensor(self.quantizer_2.model, calib_data,
                                              op_list=[('sg_mkldnn_conv_bn_act_0_output', 'CONV'),
                                                       ('data', 'input')],
                                              iteration_list=[0, 2, 4])
            inspected_qtensor = inspect_tensor(qmodel, calib_data,
                                               op_list=[('quantized_sg_mkldnn_conv_bn_act_0_output', 'CONV')],
                                               iteration_list=[0])

            self.assertNotEqual(len(inspected_tensor), 0)
            self.assertNotEqual(len(inspected_qtensor), 0)
            self.assertIsInstance(qmodel.model[0], mx.symbol.Symbol)

    def test_gluon_model(self):
        """
        Use gluon model to test gluon related functions in mxnet adaptor.
        """
        # create gluon model
        net = nn.HybridSequential()
        net.add(nn.Dense(128, activation="relu"))
        net.add(nn.Dense(64, activation="relu"))
        net.add(nn.Dense(10))
        net.initialize()

        class Quant_dataloader():
            def __init__(self, dataset, batch_size=1):
                self.dataset = dataset
                self.batch_size = batch_size

            def __iter__(self):
                for data, label in self.dataset:
                    yield data, label

            def __getitem__(self):
                pass

        valid_dataset = mx.gluon.data.vision.datasets.FashionMNIST(train=False)
        q_dataloader = Quant_dataloader(valid_dataset)
        self.quantizer_1.model = common.Model(net)
        self.quantizer_1.calib_dataloader = q_dataloader
        self.quantizer_1.eval_func = eval_func
        qmodel = self.quantizer_1()
        self.assertIsInstance(qmodel.model, mx.gluon.HybridBlock)

if __name__ == "__main__":
    unittest.main()
