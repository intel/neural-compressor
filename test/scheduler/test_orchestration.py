import copy
import os
import shutil
import unittest

import torch
import torch.nn as nn
import torchvision

from neural_compressor.config import (
    DistillationConfig,
    KnowledgeDistillationLossConfig,
    QuantizationAwareTrainingConfig,
    WeightPruningConfig,
)
from neural_compressor.data import Datasets
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.training import prepare_compression


class TestPruning(unittest.TestCase):
    model = torchvision.models.resnet18()

    def test_distillation_prune_qat_oneshot_with_new_API(self):
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(16, 3, 224, 224), low=0.0, high=1.0, label=True)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        model = copy.deepcopy(self.model)
        distillation_criterion = KnowledgeDistillationLossConfig(loss_types=["CE", "KL"])
        d_conf = DistillationConfig(copy.deepcopy(self.model), distillation_criterion)
        p_conf = WeightPruningConfig([{"start_step": 0, "end_step": 2}], target_sparsity=0.64, pruning_scope="local")
        q_conf = QuantizationAwareTrainingConfig()
        compression_manager = prepare_compression(model=model, confs=[d_conf, p_conf, q_conf])
        compression_manager.callbacks.on_train_begin()
        model = compression_manager.model

        def train_func_for_nc(model):
            epochs = 3
            iters = 3
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.1, nesterov=True, weight_decay=0.001)
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                compression_manager.callbacks.on_epoch_begin(nepoch)
                for image, target in dummy_dataloader:
                    compression_manager.callbacks.on_step_begin(cnt)
                    print(".", end="")
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
        print(20 * "=" + "test_distillation_prune_qat_oneshot" + 20 * "=")
        try:
            conv_weight = dict(model.model.layer1.named_modules())["0.conv1"].weight().dequantize()
        except:
            conv_weight = dict(model.model.layer1.named_modules())["0.conv1"].weight()
        self.assertAlmostEqual((conv_weight == 0).sum().item() / conv_weight.numel(), 0.64, delta=0.05)
        self.assertTrue("quantized" in str(type(dict(model.model.layer1.named_modules())["0.conv1"])))
        self.assertEqual(
            str(compression_manager.callbacks.callbacks_list),
            "[Distillation Callbacks, Pruning Callbacks, Quantization Aware Training Callbacks]",
        )


if __name__ == "__main__":
    unittest.main()
