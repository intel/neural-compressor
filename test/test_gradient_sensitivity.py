import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from neural_compressor.experimental.data.datasets.dummy_dataset import DummyDataset
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from transformers import BertForSequenceClassification
from neural_compressor.data import DATASETS

def build_fake_yaml():
    fake_yaml = """
        model:
          name: gradient_sensitivity_prune
          framework: pytorch
        pruning:
          approach:
            weight_compression:
              start_epoch: 0
              end_epoch: 1
              pruners:
                - !Pruner
                    start_epoch: 0
                    end_epoch: 1
                    prune_type: gradient_sensitivity
                    update_frequency: 1
                    names: [
                             'bert.encoder.layer.*.attention.self.query.weight',
                             'bert.encoder.layer.*.attention.self.query.bias',
                             'bert.encoder.layer.*.attention.self.key.weight',
                             'bert.encoder.layer.*.attention.self.key.bias',
                             'bert.encoder.layer.*.attention.self.value.weight',
                             'bert.encoder.layer.*.attention.self.value.bias',
                           ]
                    parameters: {
                                  target: 8,
                                  normalize: True,
                                  stride: 64,
                                  transpose: False,
                                  importance_inputs: ['head_mask'],
                                  importance_metric: abs_gradient,
                                }

                - !Pruner
                    start_epoch: 0
                    end_epoch: 1
                    prune_type: gradient_sensitivity
                    update_frequency: 1
                    names: [
                             'bert.encoder.layer.*.attention.output.dense.weight',
                           ]
                    parameters: {
                                  target: 8,
                                  normalize: True,
                                  stride: 64,
                                  transpose: True,
                                  importance_inputs: ['head_mask'],
                                  importance_metric: abs_gradient,
                                }

                - !Pruner
                    prune_type: gradient_sensitivity
                    names: [
                             'bert.encoder.layer.*.intermediate.dense.weight',
                             'bert.encoder.layer.*.intermediate.dense.bias',
                           ]
                    parameters: {
                                  target: 600,
                                  normalize: False,
                                  stride: 1,
                                  transpose: False,
                                  importance_inputs: [
                                               'bert.encoder.layer.*.intermediate.dense.weight',
                                               'bert.encoder.layer.*.intermediate.dense.bias',
                                              ],
                                  importance_metric: 'weighted_gradient',
                                }

                - !Pruner
                    prune_type: gradient_sensitivity
                    names: [
                             'bert.encoder.layer.*.output.dense.weight',
                           ]
                    parameters: {
                                  target: 600,
                                  normalize: False,
                                  stride: 1,
                                  transpose: True,
                                  importance_inputs: [
                                               'bert.encoder.layer.*.intermediate.dense.weight',
                                               'bert.encoder.layer.*.intermediate.dense.bias',
                                              ],
                                  importance_metric: 'weighted_gradient',
                                }

        tuning:
          accuracy_criterion:
            relative: 0.1                             # only verifying workflow, accuracy loss percentage: 10%
          exit_policy:
            timeout: 0                                   # tuning timeout (seconds)
          random_seed: 9527                            # random seed
    """
    with open('fake.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

def build_fake_yaml_unstructured():
    fake_yaml_unstructured = """
    model:
      name: imagenet_prune
      framework: pytorch

    pruning:
      approach:
        weight_compression:
          initial_sparsity: 0.0
          start_epoch: 0
          end_epoch: 4
          pruners:
            - !Pruner
                start_epoch: 1
                end_epoch: 3
                target_sparsity: 0.8
                prune_type: gradient_sensitivity
                names: ['layer1.0.conv1.weight']

            - !Pruner
                target_sparsity: 0.6
                prune_type: basic_magnitude
                update_frequency: 2
                names: ['layer1.0.conv2.weight']
    evaluation:
      accuracy:
        metric:
          topk: 1
    """
    with open('fake_unstructured.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml_unstructured)

class TestGradientSensitivity(unittest.TestCase):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    @classmethod
    def setUpClass(cls):
        build_fake_yaml()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_gradient_sensitivity(self):
        from neural_compressor.experimental import Pruning, common
        prune = Pruning('fake.yaml')

        def training_func_for_nc(model):
            inputs = {'input_ids': torch.rand([1,12]).long(),
                      'attention_mask': torch.rand([1,12]).long(),
                      'labels': torch.tensor([1]).long()}
            model.eval()

            # To calculate head prune
            prune.on_epoch_begin(0)
            head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads)
            head_mask.requires_grad_(requires_grad=True)

            outputs = model(output_attentions=True, **inputs, head_mask=head_mask)
            tmp_eval_loss, logits = outputs[:2]
            tmp_eval_loss.backward()
            prune.on_batch_end()
            prune.on_epoch_end()

        def eval_func_for_nc(model):
            pass

        prune.model = common.Model(self.model)
        prune.pruning_func = training_func_for_nc
        prune.eval_func = eval_func_for_nc
        _ = prune()
        for bertlayer in self.model.bert.encoder.layer:
            self.assertEqual(bertlayer.attention.self.query.weight.shape, (512, 768))
            self.assertEqual(bertlayer.attention.self.key.weight.shape, (512, 768))
            self.assertEqual(bertlayer.attention.self.value.weight.shape, (512, 768))
            self.assertEqual(bertlayer.attention.output.dense.weight.shape, (768, 512))
            self.assertEqual(bertlayer.intermediate.dense.weight.shape, (600, 768))
            self.assertEqual(bertlayer.output.dense.weight.shape, (768, 600))

class TestGradientSensitivityUnstructured(unittest.TestCase):
    cv_model = torchvision.models.resnet18()

    @classmethod
    def setUpClass(cls):
        build_fake_yaml_unstructured()

    @classmethod
    def tearDownClass(cls):
        os.remove('fake_unstructured.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_unstructured_pruning(self):
        from neural_compressor.experimental import Pruning, common
        prune_cv = Pruning('fake_unstructured.yaml')
        datasets = DATASETS('pytorch')
        dummy_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        def training_func_for_cv(model):
            epochs = 5
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            prune_cv.pre_epoch_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                prune_cv.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    prune_cv.on_batch_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    output = model(image)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    prune_cv.on_batch_end()
                    if cnt >= iters:
                        break
                prune_cv.on_epoch_end()
            prune_cv.post_epoch_end()
        prune_cv.model = common.Model(self.cv_model)
        prune_cv.pruning_func = training_func_for_cv
        prune_cv.eval_dataloader = dummy_dataloader
        prune_cv.train_dataloader = dummy_dataloader
        _ = prune_cv()

        # assert sparsity ratio
        conv1_weight = self.cv_model.layer1[0].conv1.weight
        conv2_weight = self.cv_model.layer1[0].conv2.weight
        self.assertAlmostEqual((conv1_weight == 0).sum().item() / conv1_weight.numel(),
                               0.8,
                               delta=0.01)
        self.assertAlmostEqual((conv2_weight == 0).sum().item() / conv2_weight.numel(),
                               0.6,
                               delta=0.01)

if __name__ == "__main__":
    unittest.main()
