import os
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn

from lpot.experimental.data.datasets.dummy_dataset import DummyDataset
from lpot.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from transformers import BertForSequenceClassification

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
        from lpot.experimental import Pruning, common
        prune = Pruning('fake.yaml')

        def training_func_for_lpot(model):
            inputs = {'input_ids': torch.rand([1,12]).long(),
                      'attention_mask': torch.rand([1,12]).long(),
                      'labels': torch.tensor([1]).long()}
            model.eval()

            # To calculate head prune
            head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads)
            head_mask.requires_grad_(requires_grad=True)

            outputs = model(output_attentions=True, **inputs, head_mask=head_mask)
            tmp_eval_loss, logits = outputs[:2]
            tmp_eval_loss.backward()
            prune.on_batch_end()
            prune.on_epoch_end()

        def eval_func_for_lpot(model):
            pass

        prune.model = common.Model(self.model)
        prune.pruning_func = training_func_for_lpot
        prune.eval_func = eval_func_for_lpot
        _ = prune()
        for bertlayer in self.model.bert.encoder.layer:
            self.assertEqual(bertlayer.attention.self.query.weight.shape, (512, 768))
            self.assertEqual(bertlayer.attention.self.key.weight.shape, (512, 768))
            self.assertEqual(bertlayer.attention.self.value.weight.shape, (512, 768))
            self.assertEqual(bertlayer.attention.output.dense.weight.shape, (768, 512))
            self.assertEqual(bertlayer.intermediate.dense.weight.shape, (600, 768))
            self.assertEqual(bertlayer.output.dense.weight.shape, (768, 600))

if __name__ == "__main__":
    unittest.main()
