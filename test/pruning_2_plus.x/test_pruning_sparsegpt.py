# import unittest

# import torch
# import torchvision
# import torch.nn as nn
# import sys
# sys.path.insert(0, './neural-compressor/')
# from neural_compressor.data import Datasets
# from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
# from neural_compressor import WeightPruningConfig
# from transformers import (AutoModelForCausalLM)


# class TestPruning(unittest.TestCase):
#     # model = torchvision.models.vit_b_16()
#     # model = AutoModelForCausalLM.from_pretrained(
#     #         "facebook/opt-125m",
#     #         )
#     def test_pruning_basic(self):
#         local_configs = [
#             {
#                 "op_names": ['encoder_layer_1.mlp*'],
#                 "target_sparsity": 0.65,
#                 "pattern": '1x1',
#                 "pruning_type": "sparse_gpt",
#                 "pruning_op_types": "Linear",
#             },
#             {
#                 "op_names": ['encoder_layer_2.mlp*'],
#                 "target_sparsity": 0.5,
#                 "pattern": '2:4',
#                 "pruning_op_types": "Linear",
#                 "pruning_type": "sparse_gpt",
#             },
#         ]
#         config = WeightPruningConfig(
#             local_configs,
#             target_sparsity=0.8,
#             start_step=1,
#             end_step=10
#         )

#         criterion = nn.CrossEntropyLoss()
#         from neural_compressor.compression.pruner import prepare_pruning
#         datasets = Datasets('pytorch')
#         # self.model.config.model_type = 'VisionTransformer.vit_b_16'
#         dummy_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
#         # dummy_dataloader = PyTorchDataLoader(dummy_dataset)
#         from torch.utils.data import DataLoader
#         dummy_dataloader = DataLoader(dummy_dataset)
#         pruning = prepare_pruning(config, self.model, dummy_dataloader, loss_func=criterion, device='cpu')


# if __name__ == "__main__":
#     unittest.main()



