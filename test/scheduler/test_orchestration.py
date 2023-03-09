import os
import copy
import shutil
import unittest

import torch
import torchvision
import torch.nn as nn
from packaging.version import Version
import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor.config import DistillationConfig, KnowledgeDistillationLossConfig
from neural_compressor.config import WeightPruningConfig
from neural_compressor.data import Datasets
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.training import prepare_compression

PT_VERSION = nc_torch.get_torch_version()
if PT_VERSION >= Version("1.8.0-rc1"):
    FX_MODE = True
else:
    FX_MODE = False

class TestPruning(unittest.TestCase):
    model = torchvision.models.resnet18()
    q_model = torchvision.models.quantization.resnet18()
    q_model.fuse_model()

    def test_distillation_prune_oneshot_with_new_API(self):
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(16, 3, 224, 224), low=0., high=1., label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        model = copy.deepcopy(self.model)
        distillation_criterion = KnowledgeDistillationLossConfig(loss_types=['CE', 'KL'])
        d_conf = DistillationConfig(copy.deepcopy(self.model), distillation_criterion)
        p_conf = WeightPruningConfig(
            [{'start_step': 0, 'end_step': 2}], target_sparsity=0.64, pruning_scope="local")
        compression_manager = prepare_compression(model=model, confs=[d_conf, p_conf])
        def train_func_for_nc(model):
            epochs = 3
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=0.001,
                                        momentum=0.1,
                                        nesterov=True,
                                        weight_decay=0.001)
            compression_manager.callbacks.on_train_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                compression_manager.callbacks.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    compression_manager.callbacks.on_step_begin(cnt)
                    print('.', end='')
                    cnt += 1
                    image = image.to(device)
                    target = target.to(device)
                    output = model(image)
                    loss = criterion(output, target)
                    loss = compression_manager.callbacks.on_after_compute_loss(image, output, loss)
                    optimizer.zero_grad()
                    loss.backward()
                    compression_manager.callbacks.on_before_optimizer_step()
                    optimizer.step()
                    compression_manager.callbacks.on_after_optimizer_step()
                    compression_manager.callbacks.on_step_end()
                    if cnt >= iters:
                        break
                compression_manager.callbacks.on_epoch_end()
            compression_manager.callbacks.on_train_end()
            return model

        train_func_for_nc(model)
        print(20 * '=' + 'test_distillation_prune_oneshot' + 20 * '=')
        try:
            conv_weight = model.layer1[0].conv1.weight().dequantize()
        except:
            conv_weight = model.layer1[0].conv1.weight
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(),
                               0.64,
                               delta=0.05)
        self.assertEqual(
            str(compression_manager.callbacks.callbacks_list),
            "[Distillation Callbacks, Pruning Callbacks]"
        )


if __name__ == "__main__":
    unittest.main()