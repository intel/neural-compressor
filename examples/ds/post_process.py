import json
from loguru import logger

quantization_config = {
    "_json_file": "/tmp/tmpe3ckugb_.json",
    "allowlist": {
        "names": [],
        "types": [
            "Matmul",
            "Linear",
            "ParallelLMHead",
            "RowParallelLinear",
            "ColumnParallelLinear",
            "MergedColumnParallelLinear",
            "QKVParallelLinear",
            "FalconLinear",
            "KVCache",
            "VLLMKVCache",
            "Conv2d",
            "LoRACompatibleLinear",
            "LoRACompatibleConv",
            "Softmax",
            "ModuleFusedSDPA",
            "MoeMatmul",
            "ReplicatedLinear",
            "FusedMoE",
            "GaudiMixtralSparseMoeBlock",
            "VllmMixtureOfExpertsOp",
            "LinearLayer",
            "LinearAllreduce",
            "ScopedLinearAllReduce",
            "LmHeadLinearAllreduce",
        ],
    },
    "blocklist": {},
    "dump_stats_path": "./hqt_output/measure",
    "fake_quant": "False",
    "fp8_config": "E4M3",
    "hp_dtype": "bf16",
    "measure_on_hpu": True,
    "mod_dict": {},
    "mode": "LOAD",
    "observer": "maxabs",
    "scale_format": "const",
    "scale_method": "maxabs_hw",
    "scale_params": {},
    "use_qdq": "False",
}


# add the quantization config to config.json
def update_config(model_path, qmodel_path):
    import json
    import os

    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
        config["quantization_config"] = quantization_config
        logger.info(f"Updated config: {config}")
        config_filepath = os.path.join(qmodel_path, "config.json")
        logger.debug(f"Saving config to {config_filepath}")
        with open(config_filepath, "w") as f:
            json.dump(config, f, indent=4)


MODEL_FILE_LST = [
    "configuration_deepseek.py",
    "generation_config.json",
    "modeling_deepseek.py",
    "tokenizer.json",
    "tokenizer_config.json",
]


def cp_model_files(model_path, qmodel_path):
    # copy model files
    import shutil
    import os

    for file in MODEL_FILE_LST:
        logger.debug(f"Copying {file} from {model_path} to {qmodel_path}")
        file_path = os.path.join(model_path, file)
        # check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist")
            raise FileNotFoundError(f"File {file_path} does not exist")
        shutil.copy(os.path.join(model_path, file), qmodel_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--qmodel_path", type=str, required=True)
    parser.add_argument("--low_cpu_mem", action="store_true", help="Load weight file one by one to reduce memory usage")
    args = parser.parse_args()
    # update the config
    update_config(args.model_path, args.qmodel_path)
    # copy model files
    cp_model_files(args.model_path, args.qmodel_path)
