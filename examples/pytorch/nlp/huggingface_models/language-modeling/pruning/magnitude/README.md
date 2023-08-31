Step-by-Step
============

# single GPU

```
bash run.sh
```

# multi GPU

we use `accelerate` and `deepspeed ZeRO Stage-2` to conduct weight magnitude pruning

### Accelerate DeepSpeed Plugin

On your machine(s) just run:
```
accelerate config
```

and answer the questions asked. It will ask whether you want to use a config file for DeepSpeed to which you should answer no. Then answer the following questions to generate a basic DeepSpeed config. This will generate a config file that will be used automatically to properly set the default options when doing

For instance,

```
compute_environment: LOCAL_MACHINE
deepspeed_config:
 deepspeed_config_file: config/zero_stage2_config.json
 zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```
with the contents of `config/zero_stage2_config.json` being:

```
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true,
    "min_loss_scale": 1,
    "opt_level": "O2"
  },
  "zero_optimization": {
    "stage": 2,
    "offload_param": {
      "device": "cpu"
    },
    "offload_optimizer": {
      "device": "cpu"
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "torch_adam": true,
      "adam_w_mode": true
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0.0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto",
      "total_num_steps": "auto",
      "warmup_type": "cosine"
    }
  }
}
```

### pruning

```
bash run_ds.sh 
```
