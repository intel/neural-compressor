import argparse
import os
import platform
import re
from typing import Optional, Union

import psutil

system = platform.system()
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--framework", type=str, required=True, help="Framework of model.")
    parser.add_argument("--dataset_location", type=str, required=True, help="Location of dataset used for model.")
    parser.add_argument("--strategy", type=str, required=False, help="Strategy to update.")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size.")
    parser.add_argument("--new_benchmark", type=str, required=False, help="Whether to modify benchmark config.")
    parser.add_argument("--multi_instance", type=str, required=False, help="Whether to eval in multi-instance.")
    return parser.parse_args()


def update_yaml_dataset(yaml, framework, dataset_location):
    if not os.path.isfile(yaml):
        raise Exception(f"Not found yaml config at '{yaml}' location.")

    print("Reading config")
    with open(yaml, "r") as config:
        lines = config.readlines()

    # Update dataset
    if framework != "pytorch":
        val_txt_location = os.path.dirname(dataset_location) + f"{os.path.sep}" + "val.txt"

        patterns = {
            "root_path": {
                "pattern": r"root:.*/path/to/(calibration|evaluation)/dataset/?",
                "replacement": f"root: {dataset_location}",
            },
            "data_path": {
                "pattern": r"data_path:.*/path/to/(calibration|evaluation)/dataset/?",
                "replacement": f"data_path: {dataset_location}",
            },
            "image_list": {
                "pattern": r"image_list:.*/path/to/(calibration|evaluation)/label/?",
                "replacement": f"image_list: {val_txt_location}",
            },
            "data_dir": {
                "pattern": r"data_dir:.*/path/to/dataset/?",
                "replacement": f"data_dir: {dataset_location}",
            },
        }
        print("======= update_yaml_dataset =======")
        with open(yaml, "w") as config:
            for line in lines:
                for key, key_patterns in patterns.items():
                    if re.search(key_patterns["pattern"], line):
                        print(f"Replacing {key} key.")
                        line = re.sub(key_patterns["pattern"], key_patterns["replacement"], line)
                config.write(line)

    else:
        val_dataset = dataset_location + f"{os.path.sep}" + "val"
        train_dataset = dataset_location + f"{os.path.sep}" + "train"
        patterns = {
            "calibration_dataset": {
                "pattern": r"root:.*/path/to/calibration/dataset/?",
                "replacement": f"root: {train_dataset}",
            },
            "evaluation_dataset": {
                "pattern": r"root:.*/path/to/evaluation/dataset/?",
                "replacement": f"root: {val_dataset}",
            },
        }

        print("======= update_yaml_dataset =======")
        with open(yaml, "w") as config:
            for line in lines:
                for key, key_patterns in patterns.items():
                    if re.search(key_patterns["pattern"], line):
                        print(f"Replacing {key} key.")
                        line = re.sub(key_patterns["pattern"], key_patterns["replacement"], line)
                config.write(line)


def update_yaml_config_tuning(
    yaml_file,
    strategy=None,
    mode=None,
    batch_size=None,
    iteration=None,
    max_trials=None,
    algorithm=None,
    timeout=None,
    strategy_token=None,
    sampling_size=None,
    dtype=None,
    tf_new_api=None,
):
    with open(yaml_file) as f:
        yaml_config = yaml.round_trip_load(f, preserve_quotes=True)

    if algorithm:
        try:
            model_wise = yaml_config.get("quantization", {}).get("model_wise", {})
            prev_activation = model_wise.get("activation", {})
            if not prev_activation:
                model_wise.update({"activation": {}})
                prev_activation = model_wise.get("activation", {})
            prev_activation.update({"algorithm": algorithm})
        except Exception as e:
            print(f"[ WARNING ] {e}")

    if timeout:
        try:
            exit_policy = yaml_config.get("tuning", {}).get("exit_policy", {})
            prev_timeout = exit_policy.get("timeout", None)
            exit_policy.update({"timeout": timeout})
            print(f"Changed {prev_timeout} to {timeout}")
        except Exception as e:
            print(f"[ WARNING ] {e}")

    if strategy and strategy != "basic":  # Workaround for PyTorch huggingface models (`sed` in run_quant.sh)
        try:
            tuning_config = yaml_config.get("tuning", {})
            prev_strategy = tuning_config.get("strategy", {})
            if not prev_strategy:
                tuning_config.update({"strategy": {}})
                prev_strategy = tuning_config.get("strategy", {})
            strategy_name = prev_strategy.get("name", None)
            prev_strategy.update({"name": strategy})
            if strategy == "sigopt":
                prev_strategy.update(
                    {
                        "sigopt_api_token": strategy_token,
                        "sigopt_project_id": "lpot",
                        "sigopt_experiment_name": "lpot-tune",
                    }
                )
            if strategy == "hawq":
                prev_strategy.update({"loss": "CrossEntropyLoss"})
            print(f"Changed {strategy_name} to {strategy}")
        except Exception as e:
            print(f"[ WARNING ] {e}")

    if max_trials and max_trials > 0:
        try:
            tuning_config = yaml_config.get("tuning", {})
            prev_exit_policy = tuning_config.get("exit_policy", {})
            if not prev_exit_policy:
                tuning_config.update({"exit_policy": {"max_trials": max_trials}})
            else:
                prev_max_trials = prev_exit_policy.get("max_trials", None)
                prev_exit_policy.update({"max_trials": max_trials})
                print(f"Changed {prev_max_trials} to {max_trials}")
        except Exception as e:
            print(f"[ WARNING ] {e}")

    if mode == "accuracy":
        try:
            # delete performance part in yaml if exist
            performance = yaml_config.get("evaluation", {}).get("performance", {})
            if performance:
                yaml_config.get("evaluation", {}).pop("performance", {})
            # accuracy batch_size replace
            if batch_size:
                try:
                    dataloader = yaml_config.get("evaluation", {}).get("accuracy", {}).get("dataloader", {})
                    prev_batch_size = dataloader.get("batch_size", None)
                    dataloader.update({"batch_size": batch_size})
                    print(f"Changed accuracy batch size from {prev_batch_size} to {batch_size}")
                except Exception as e:
                    print(f"[ WARNING ] {e}")
        except Exception as e:
            print(f"[ WARNING ] {e}")
    elif mode:
        try:
            # delete accuracy part in yaml if exist
            accuracy = yaml_config.get("evaluation", {}).get("accuracy", {})
            if accuracy:
                yaml_config.get("evaluation", {}).pop("accuracy", {})
            # performance iteration replace
            if iteration:
                try:
                    performance = yaml_config.get("evaluation", {}).get("performance", {})
                    prev_iteration = performance.get("iteration", None)
                    performance.update({"iteration": iteration})
                    print(f"Changed performance batch size from {prev_iteration} to {iteration}")
                except Exception as e:
                    print(f"[ WARNING ] {e}")

            if batch_size and mode == "latency":
                try:
                    dataloader = yaml_config.get("evaluation", {}).get("performance", {}).get("dataloader", {})
                    prev_batch_size = dataloader.get("batch_size", None)
                    dataloader.update({"batch_size": batch_size})
                    print(f"Changed accuracy batch size from {prev_batch_size} to {batch_size}")
                except Exception as e:
                    print(f"[ WARNING ] {e}")

        except Exception as e:
            print(f"[ WARNING ] {e}")

    if sampling_size:
        try:
            calibration = yaml_config.get("quantization", {}).get("calibration", {})
            prev_sampling_size = calibration.get("sampling_size", None)
            calibration.update({"sampling_size": sampling_size})
            print(f"Changed calibration sampling size from {prev_sampling_size} to {sampling_size}")
        except Exception as e:
            print(f"[ WARNING ] {e}")

    if dtype:
        try:
            quantization = yaml_config.get("quantization", {})
            prev_dtype = quantization.get("dtype", None)
            quantization.update({"dtype": dtype})
            print(f"Changed dtype from {prev_dtype} to {dtype}")
        except Exception as e:
            print(f"[ WARNING ] {e}")

    if tf_new_api == "true":
        try:
            model = yaml_config.get("model", {})
            prev_framework = model.get("framework", None)
            model.update({"framework": "inteltensorflow"})
            print(f"Changed framework from {prev_framework} to inteltensorflow")
        except Exception as e:
            print(f"[ WARNING ] {e}")

    print("====== update_yaml_config_tuning ========")

    yaml_content = yaml.round_trip_dump(yaml_config)

    with open(yaml_file, "w") as output_file:
        output_file.write(yaml_content)


def update_yaml_config_benchmark_acc(yaml_path: str, batch_size=None):
    with open(yaml_path) as f:
        yaml_config = yaml.round_trip_load(f, preserve_quotes=True)
    try:
        accuracy = yaml_config.get("evaluation", {}).get("accuracy", {})
        if not accuracy:
            raise AttributeError
        dataloader = accuracy.get("dataloader", {})
        if dataloader:
            dataloader.update({"batch_size": batch_size})
        configs = accuracy.get("configs", {})
        if configs:
            del accuracy["configs"]
    except Exception as e:
        print(f"[ WARNING ] {e}")

    print("====== update_yaml_config_benchmark_acc ========")

    yaml_content = yaml.round_trip_dump(yaml_config)

    with open(yaml_path, "w") as output_file:
        output_file.write(yaml_content)


def update_yaml_config_benchmark_perf(yaml_path: str, batch_size=None, multi_instance=None):
    # Get cpu information for multi-instance
    total_cores = psutil.cpu_count(logical=False)
    total_sockets = 1
    ncores_per_socket = total_cores / total_sockets
    ncores_per_instance = ncores_per_socket
    iters = 100

    if multi_instance == "true":
        ncores_per_instance = 4
        iters = 500

    with open(yaml_path) as f:
        yaml_config = yaml.round_trip_load(f, preserve_quotes=True)
    try:
        performance = yaml_config.get("evaluation", {}).get("performance", {})
        if not performance:
            raise AttributeError
        dataloader = performance.get("dataloader", {})
        if dataloader:
            dataloader.update({"batch_size": batch_size})
        performance.update({"iteration": iters})
        configs = performance.get("configs", {})
        if not configs:
            raise AttributeError
        else:
            configs.update(
                {
                    "cores_per_instance": int(ncores_per_instance),
                    "num_of_instance": int(ncores_per_socket // ncores_per_instance),
                }
            )
            for attr in ["intra_num_of_threads", "inter_num_of_threads", "kmp_blocktime"]:
                if configs.get(attr):
                    del configs[attr]
            print(configs)
    except Exception as e:
        print(f"[ WARNING ] {e}")

    print("====== update_yaml_config_benchmark_perf ========")

    yaml_content = yaml.round_trip_dump(yaml_config)

    with open(yaml_path, "w") as output_file:
        output_file.write(yaml_content)


if __name__ == "__main__":
    args = parse_args()
    update_yaml_dataset(args.yaml, args.framework, args.dataset_location)
    update_yaml_config_tuning(args.yaml, strategy=args.strategy)
    print("===== multi_instance={} ====".format(args.multi_instance))
    if args.new_benchmark == "true":
        update_yaml_config_benchmark_acc(args.yaml, batch_size=args.batch_size)
        update_yaml_config_benchmark_perf(args.yaml, batch_size=args.batch_size, multi_instance=args.multi_instance)
