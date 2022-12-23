import copy
import shutil
import torch
import torch.nn as nn
import unittest
import os
from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization
from neural_compressor.experimental import Quantization
from neural_compressor.data import Datasets, DATALOADERS, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

fake_bf8_yaml = '''
    model:
        name: imagenet
        framework: pytorch_fx

    quantization:
        # per op-type auto quantization
        optimization_level: 0
        # approach not needed for fp8_e5m2
        precision: fp8_e5m2   # select from fp8_e5m2, fp8_e4m3, int8
        calibration:
            batchnorm_sampling_size: 3000

    tuning:
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            timeout: 0
        random_seed: 9527
    '''


fake_dyn_yaml = '''
    model:
        name: imagenet
        framework: pytorch_fx

    quantization:
        optimization_level: 0
        approach: post_training_dynamic_quant
        precision: fp8_e3m4
        calibration:
            batchnorm_sampling_size: 3000

    tuning:
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            timeout: 0
        random_seed: 9527
    '''


fake_ptq_yaml = '''
    model:
        name: imagenet
        framework: pytorch_fx

    quantization:
        optimization_level: 0
        approach: post_training_static_quant
        precision: fp8_e4m3
        calibration:
            batchnorm_sampling_size: 3000
            sampling_size: 300

    tuning:
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            timeout: 0
        random_seed: 9527
    '''

fake_bf8_fallback_yaml = '''
    model:
        name: imagenet
        framework: pytorch_fx

    quantization:
        # approach not needed for fp8_e5m2
        precision: fp8_e5m2
        calibration:
            batchnorm_sampling_size: 3000
        optype_wise: {
            "Embedding": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "LayerNorm": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "Matmul": {
                "activation": {"dtype": ["fp32"]},
            },
            "BatchMatmul": {
                "activation": {"dtype": ["fp32"]},
            },
        }
        op_wise: {
            "conv1": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
        }

    tuning:
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            timeout: 0
        random_seed: 9527
    '''

fake_dyn_fallback_yaml = '''
    model:
        name: imagenet
        framework: pytorch_fx

    quantization:
        approach: post_training_dynamic_quant
        precision: fp8_e4m3
        calibration:
            batchnorm_sampling_size: 3000
        optype_wise: {
            "Embedding": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "LayerNorm": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "Matmul": {
                "activation": {"dtype": ["fp32"]},
            },
            "BatchMatmul": {
                "activation": {"dtype": ["fp32"]},
            },
        }
        op_wise: {
            "conv1": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
        }

    tuning:
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            timeout: 0
        random_seed: 9527
    '''

fake_ptq_fallback_yaml = '''
    model:
        name: imagenet
        framework: pytorch_fx

    quantization:
        approach: post_training_static_quant
        precision: fp8_e3m4
        calibration:
            batchnorm_sampling_size: 3000
            sampling_size: 300
        optype_wise: {
            "Embedding": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "LayerNorm": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "Matmul": {
                "activation": {"dtype": ["fp32"]},
            },
            "BatchMatmul": {
                "activation": {"dtype": ["fp32"]},
            },
        }
        op_wise: {
            "conv1": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
        }

    tuning:
        accuracy_criterion:
            relative:  0.01
        exit_policy:
            timeout: 0
        random_seed: 9527
    '''


def build_pytorch_yaml():
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_ptq_yaml)

    with open('dynamic_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_dyn_yaml)

    with open('bf8_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_bf8_yaml)

    with open('ptq_fallback_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_ptq_fallback_yaml)

    with open('dynamic_fallback_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_dyn_fallback_yaml)

    with open('bf8_fallback_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_bf8_fallback_yaml)


class DummyNLPDataloader(object):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b, return_tensors='pt')
        self.batch_size = 1

    def __iter__(self):
        yield self.encoded_dict

    def __next__(self):
        return self.encoded_dict


i = 0
def eval_func(model):
    global i
    i += 1
    if i <= 3:
        return 1
    else:
        return 0

def eval_func1(model):
    global i
    i += 1
    if i == 1 or i > 2:
        return 1
    else:
        return 0

def eval_func2(model):
    global i
    i += 1
    if i == 1 or (i > 2 and i != 5):
        return 1
    else:
        return 0


class TestPytorchFP8Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from torchvision.models import resnet18
        self.cv_model = resnet18()
        self.cv_dataset = Datasets("pytorch")["dummy"]((10, 3, 224, 224))
        self.cv_dataloader = DATALOADERS["pytorch"](self.cv_dataset)
        self.nlp_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.nlp_dataloader = DummyNLPDataloader(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        build_pytorch_yaml()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_CV_quantization_new_API(self):
        model = self.cv_model
        quant_conf = PostTrainingQuantConfig(
            precision="fp8_e5m2",
            # by default, calibration_sampling_size=100, batchnorm_calibration_sampling_size = 3000
            calibration_sampling_size=[300],
            batchnorm_calibration_sampling_size=[3000],
        )
        global i
        i = 0
        q_model = quantization.fit(
            model,
            quant_conf,
            eval_func = eval_func1,
            calib_dataloader=self.cv_dataloader)

        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic", 
                    precision="fp8_e4m3",
                    calibration_sampling_size=[300],
                    batchnorm_calibration_sampling_size=[3000],
                )
            elif fake_yaml == "static":
                # static contains minmax, kl algo.
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e4m3",
                    calibration_sampling_size=[300],
                    batchnorm_calibration_sampling_size=[3000],
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func = eval_func1,
                calib_dataloader=self.cv_dataloader)

        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic", 
                    precision="fp8_e3m4",
                    calibration_sampling_size=[300],
                    batchnorm_calibration_sampling_size=[3000],
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e3m4",
                    calibration_sampling_size=[300],
                    batchnorm_calibration_sampling_size=[3000],
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func = eval_func1,
                calib_dataloader=self.cv_dataloader)
            q_model.save("./saved")

    def test_CV_quantization_new_API_quant_level_0(self):
        model = self.cv_model
        global i
        i = 0
        quant_conf = PostTrainingQuantConfig(optimization_level=0, precision="fp8_e5m2")
        q_model = quantization.fit(
            model,
            quant_conf,
            eval_func=eval_func,
            calib_dataloader=self.cv_dataloader)

        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="dynamic", precision="fp8_e4m3")
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="static", precision="fp8_e4m3")
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func=eval_func,
                calib_dataloader=self.cv_dataloader)

        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="dynamic", precision="fp8_e3m4")
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="static", precision="fp8_e3m4")
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func=eval_func,
                calib_dataloader=self.cv_dataloader)

    def test_CV_quantization_new_API_fallback(self):
        op_type_list={
            "Linear": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            }
        }
        op_name_list={
            "conv1": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            }
        }
        model = self.cv_model
        quant_conf = PostTrainingQuantConfig(
            precision="fp8_e5m2",
            op_type_list=op_type_list,
            op_name_list=op_name_list,
        )
        q_model = quantization.fit(
            model,
            quant_conf,
            calib_dataloader=self.cv_dataloader)
        for fake_yaml in ["dynamic", "static"]:
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e4m3",
                    op_type_list=op_type_list,
                    op_name_list=op_name_list,
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e4m3",
                    op_type_list=op_type_list,
                    op_name_list=op_name_list,
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.cv_dataloader)
        for fake_yaml in ["dynamic", "static"]:
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e3m4",
                    op_type_list=op_type_list,
                    op_name_list=op_name_list,
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e3m4",
                    op_type_list=op_type_list,
                    op_name_list=op_name_list,
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.cv_dataloader)
            q_model.save("./saved")

    def test_CV_quantization_new_API_fallback_env(self):
        # fallback first conv and last linear ops as a recipe
        os.environ['DISABLE_FIRST_CONV'] = "True"
        os.environ['DISABLE_LAST_LINEAR'] = "True"
        # Use environment variables to control op_types.
        os.environ['FP8_OP_TYPE_LIST']="['linear', 'conv2d', \
                                        'bmm', 'amm', 'mm', \
                                        'add', 'mul', 'div']"
        model = self.cv_model
        quant_conf = PostTrainingQuantConfig(
            precision="fp8_e5m2",
        )
        q_model = quantization.fit(
            model,
            quant_conf,
            calib_dataloader=self.cv_dataloader)
        for fake_yaml in ["dynamic", "static"]:
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e4m3",
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e4m3",
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.cv_dataloader)
        for fake_yaml in ["dynamic", "static"]:
            model = self.cv_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e3m4",
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e3m4",
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.cv_dataloader)
            q_model.save("./saved")

    def test_NLP_quantization_new_API(self):
        model = self.nlp_model
        quant_conf = PostTrainingQuantConfig(precision="fp8_e5m2")
        global i
        i = 0
        q_model = quantization.fit(
            model,
            quant_conf,
            eval_func=eval_func2,
            calib_dataloader=self.nlp_dataloader)

        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(approach="dynamic", precision="fp8_e4m3",)
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(approach="static", precision="fp8_e4m3",)
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func = eval_func2,
                calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None
            )
        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(approach="dynamic", precision="fp8_e3m4",)
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(approach="static", precision="fp8_e3m4",)
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func = eval_func2,
                calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None
            )
        q_model.save("./saved")

    def test_NLP_quantization_new_API_quant_level_0(self):
        model = self.nlp_model
        global i
        i = 0
        quant_conf = PostTrainingQuantConfig(optimization_level=0, precision="fp8_e5m2")
        q_model = quantization.fit(
            model,
            quant_conf,
            eval_func=eval_func,
            calib_dataloader=self.nlp_dataloader)

        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="dynamic", precision="fp8_e4m3")
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="static", precision="fp8_e4m3")
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func=eval_func,
                calib_dataloader=self.nlp_dataloader)

        for fake_yaml in ["dynamic", "static"]:
            i = 0
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="dynamic", precision="fp8_e3m4")
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(optimization_level=0, approach="static", precision="fp8_e3m4")
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func=eval_func,
                calib_dataloader=self.nlp_dataloader)

    def test_NLP_quantization_new_API_fallback(self):
        op_type_list={
            "Embedding": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "LayerNorm": {
                "activation": {"dtype": ["fp32"]},
                "weight": {"dtype": ["fp32"]},
            },
            "Matmul": {
                "activation": {"dtype": ["fp32"]},
            },
            "BatchMatmul": {
                "activation": {"dtype": ["fp32"]},
            },
        }
        model = self.nlp_model
        quant_conf = PostTrainingQuantConfig(
            approach="dynamic",
            precision="fp8_e5m2",
            op_type_list=op_type_list,
        )
        q_model = quantization.fit(
            model,
            quant_conf,
            calib_dataloader=self.nlp_dataloader)

        for fake_yaml in ["static"]:
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e4m3",
                    op_type_list=op_type_list,
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e4m3",
                    op_type_list=op_type_list,
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None
            )

        for fake_yaml in ["static"]:
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e3m4",
                    op_type_list=op_type_list,
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e3m4",
                    op_type_list=op_type_list,
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None
            )
        q_model.save("./saved")


    def test_NLP_quantization_new_API_fallback_env(self):
        # Use environment variables to control op_types.
        os.environ['FP8_OP_TYPE_LIST'] = "['linear', 'conv2d', \
                                        'bmm', 'amm', 'mm', \
                                        'add', 'mul', 'div']"
        model = self.nlp_model
        quant_conf = PostTrainingQuantConfig(
            approach="dynamic",
            precision="fp8_e5m2",
        )
        q_model = quantization.fit(
            model,
            quant_conf,
            calib_dataloader=self.nlp_dataloader)

        for fake_yaml in ["static"]:
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e4m3",
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e4m3",
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None
            )

        for fake_yaml in ["static"]:
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(
                    approach="dynamic",
                    precision="fp8_e3m4",
                )
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(
                    approach="static",
                    precision="fp8_e3m4",
                )
            q_model = quantization.fit(
                model,
                quant_conf,
                calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None
            )
        q_model.save("./saved")

    def test_CV_quantization_old_API(self):
        for fake_yaml in ['dynamic_yaml.yaml', 'ptq_yaml.yaml', 'bf8_yaml.yaml', \
                          'ptq_fallback_yaml.yaml', 'dynamic_fallback_yaml.yaml', \
                          'bf8_fallback_yaml.yaml']:
            model = self.cv_model
            quantizer = Quantization(fake_yaml)
            quantizer.model = model
            quantizer.calib_dataloader = self.cv_dataloader
            q_model = quantizer.fit()
            q_model.save('./saved')

    def test_NLP_quantization_old_API(self):
        for fake_yaml in ['dynamic_yaml.yaml', 'ptq_yaml.yaml', 'bf8_yaml.yaml', \
                          'ptq_fallback_yaml.yaml', 'dynamic_fallback_yaml.yaml', \
                          'bf8_fallback_yaml.yaml']:
            model = self.nlp_model
            quantizer = Quantization(fake_yaml)
            quantizer.model = model
            quantizer.calib_dataloader = self.nlp_dataloader
            q_model = quantizer.fit()
            q_model.save('./saved')

    def test_NLP_quantization_pythonic_API(self):
        model = self.nlp_model
        from neural_compressor.conf.config import QuantConf
        from neural_compressor.experimental import Quantization
        quant_config = QuantConf()
        quant_config.usr_cfg.quantization.approach = "post_training_dynamic_quant"
        quant_config.usr_cfg.model.framework = "pytorch"
        quantizer = Quantization(quant_config)
        quantizer.model = model
        quantizer.calib_dataloader = self.nlp_dataloader
        q_model = quantizer.fit()
        q_model.save('./saved')

if __name__ == "__main__":
    unittest.main()
