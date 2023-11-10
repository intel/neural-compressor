import torch
from neural_compressor.config import WeightPruningConfig
from neural_compressor.training import prepare_compression
from neural_compressor.utils import logger


# DeepSpeed related import
import os
import json
import argparse
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from deepspeed.utils.utility import ForkedPdb
import deepspeed.comm as dist
from deepspeed.utils import (
    safe_get_full_fp32_param, 
    safe_get_full_grad, 
    safe_get_full_optimizer_state, 
    safe_set_full_fp32_param, 
    safe_set_full_optimizer_state,
    # get_local_fp32_param,
    )
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

rank = int(os.environ['RANK'])
print('seed:', 2222 + rank)
torch.random.manual_seed(2222 + rank)

def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path

def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--zero', type=int, default=3)
    parser.add_argument('--zero_hpz_partition_size', type=int, default=1)
    args = parser.parse_args()  #args=''

    config_dict["zero_optimization"]["stage"] = args.zero
    config_dict["zero_optimization"]["zero_hpz_partition_size"] = args.zero_hpz_partition_size
    print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path
    return args

config_dict = {
    "train_batch_size": 256,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
        }
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 8
    },
    "zero_optimization": {
        "stage": 0,
        "reduce_bucket_size": 20,
        "zero_hpz_partition_size": 1,
        "sub_group_size": 2000*4000,
        "reduce_scatter": True,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False
    }
}
#        "initial_scale_power": 15
args = get_args('/tmp/', config_dict)

def test_conv1_prunig():
    local_config = [
        {
            "op_names": ["conv1.*"],
            "target_sparsity": 0.6,
            "pattern": "4x1",
            "pruning_type": "snip",
            "pruning_scope": "global",
        },
        {
            "op_names": ["conv2.*"], 
            "target_sparsity": 0.5, 
            "pattern": "2:4", 
            "pruning_scope": "global"},
    ]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(4, 4, 2)
            self.act = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv1d(4, 4, 2)
            self.linear = torch.nn.Linear(32, 3)
            self.criterion = torch.nn.CrossEntropyLoss()

        def forward(self, x, y):
            # print(f"x device: {x.device}")
            out = self.conv1(x)
            out = self.act(out)
            out = self.conv2(out)
            out = out.view(1, -1)
            out = self.linear(out)
            return self.criterion(out, y)

    model = Model()
    
    model, _, _, _ = deepspeed.initialize(args=args,
                                        model=model,
                                        model_parameters=model.parameters(),
                                        dist_init_required=True)

    data = torch.rand((1, 4, 10))
    device = f"cuda:{dist.get_rank()}"
    # data = data.to(device)
    # # model.to(device)
    # # output = model(data)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    config = WeightPruningConfig(local_config, target_sparsity=0.8, start_step=1, end_step=10)
    compression_manager = prepare_compression(model=model, confs=config)
    compression_manager.callbacks.on_train_begin()
    logger.info("========== Start the tuning process ============")
    for epoch in range(2):
        logger.info(f"[EPOCH: {epoch}][PRE EPOCH] ============")
        logger.info(f"[EPOCH] ======== {epoch} ============")
        model.train()
        compression_manager.callbacks.on_epoch_begin(epoch)
        local_step = 0
        for i in range(3):
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE BATCH] ============")
            data, target = torch.rand((1, 4, 10), requires_grad=True), torch.empty(1, dtype=torch.long).random_(3)
            data = data.to(device).half()
            target = target.to(device)
            compression_manager.callbacks.on_step_begin(local_step)
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE_FORWARD]")
            loss = model(data, target)
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][AFTER_FORWARD]")
            # loss = criterion(output, target)
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][CALCULATED LOSS]")
            # optimizer.zero_grad()
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE_BACKWARD]")
            model.backward(loss)
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][AFTER_BACKWARD]")
            compression_manager.callbacks.on_before_optimizer_step()
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][PRE_OPTIMIZER_STEP]")
            model.step()
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][AFTER_OPTIMIZER_STEP]")
            compression_manager.callbacks.on_after_optimizer_step()
            compression_manager.callbacks.on_step_end()
            local_step += 1
            logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][END BATCH] ============")
        logger.info(f"[EPOCH: {epoch}][Batch: {i+1}][END EPOCH] ============")
            

        compression_manager.callbacks.on_epoch_end()
    compression_manager.callbacks.on_train_end()
    compression_manager.callbacks.on_before_eval()
    compression_manager.callbacks.on_after_eval()
    
test_conv1_prunig()