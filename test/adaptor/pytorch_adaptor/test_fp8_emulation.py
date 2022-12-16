import copy
import shutil
import torch
import torch.nn as nn
import unittest
import os
from neural_compressor import PostTrainingQuantConfig
from neural_compressor.data import DATASETS, DATALOADERS
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.model import MODELS
from neural_compressor.experimental import Quantization, common
from neural_compressor.experimental.data.datasets.dataset import DATASETS
from neural_compressor import quantization
from transformers import AutoModelForSequenceClassification, AutoTokenizer


fake_dyn_yaml = '''
    model:
      name: imagenet
      framework: pytorch_fx
      precision: fp8_e4m3

    quantization:
      approach: post_training_dynamic_quant

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
      approach: post_training_static_quant
      precision: fp8_e4m3

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
    if i == 1:
        return 1
    else:
        return 0
class TestPytorchFP8Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from torchvision.models import resnet18
        self.cv_model = resnet18()
        self.cv_dataset = DATASETS("pytorch")["dummy"]((10, 3, 224, 224))
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

    # def test_CV_quantization_new_API(self):
    #     for fake_yaml in ["static"]:
    #         model = self.cv_model
    #         if fake_yaml == "dynamic":
    #             quant_conf = PostTrainingQuantConfig(approach="dynamic")
    #         elif fake_yaml == "static":
    #             quant_conf = PostTrainingQuantConfig(approach="static")
    #         q_model = quantization.fit(
    #             model,
    #             quant_conf,
    #             eval_func=eval_func,
    #             calib_dataloader=self.cv_dataloader if fake_yaml == "static" else None)
    #         q_model.save("./saved")

    def test_NLP_quantization_new_API(self):
        for fake_yaml in ["dynamic"]:
            model = self.nlp_model
            if fake_yaml == "dynamic":
                quant_conf = PostTrainingQuantConfig(approach="dynamic")
            elif fake_yaml == "static":
                quant_conf = PostTrainingQuantConfig(approach="static")
            q_model = quantization.fit(
                model,
                quant_conf,
                eval_func=eval_func,
                calib_dataloader=self.nlp_dataloader if fake_yaml == "static" else None
            )
            q_model.save("./saved")

    # def test_CV_quantization_old_API(self):
    #     for fake_yaml in ['ptq_yaml.yaml']:
    #         model = self.cv_model
    #         quantizer = Quantization(fake_yaml)
    #         quantizer.model = model
    #         if fake_yaml == 'ptq_yaml.yaml':
    #             quantizer.calib_dataloader = self.cv_dataloader
    #         q_model = quantizer.fit()
    #         q_model.save('./saved')

    # def test_NLP_quantization_old_API(self):
    #     for fake_yaml in ['dynamic_yaml.yaml', 'ptq_yaml.yaml']:
    #         model = self.nlp_model
    #         quantizer = Quantization(fake_yaml)
    #         quantizer.model = model
    #         if fake_yaml == 'ptq_yaml.yaml':
    #             quantizer.calib_dataloader = self.nlp_dataloader
    #         q_model = quantizer.fit()
    #         q_model.save('./saved')

if __name__ == "__main__":
    unittest.main()
