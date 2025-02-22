# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import sys

import numpy as np


def fix_cache_inputs(json_data):
    for layer_index in range(len(json_data["Nodes"])):
        matmul_av_input = None
        v_cache_input = None
        matmul_qk_input = None
        k_cache_input = None

        for node_name, node_info in json_data["Nodes"].items():
            if f"model.layers.{layer_index}.self_attn.attn.impl.matmul_av" in node_name:
                matmul_av_input = node_info["inputs"][1]
            if f"model.layers.{layer_index}.self_attn.attn.impl.v_cache" in node_name:
                v_cache_input = node_info["inputs"][0]
            if f"model.layers.{layer_index}.self_attn.attn.impl.matmul_qk" in node_name:
                matmul_qk_input = node_info["inputs"][1]
            if f"model.layers.{layer_index}.self_attn.attn.impl.k_cache" in node_name:
                k_cache_input = node_info["inputs"][0]
        if matmul_av_input != v_cache_input:
            json_data["Nodes"][f"model.layers.{layer_index}.self_attn.attn.impl.matmul_av"]["inputs"][1] = v_cache_input
        if matmul_qk_input != k_cache_input:
            json_data["Nodes"][f"model.layers.{layer_index}.self_attn.attn.impl.matmul_qk"]["inputs"][1] = k_cache_input

    return json_data


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run the measurements parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--measurements",
        type=str,
        help="full path to the directory of the measurements that should be fixed",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="",
        help="path to the directory where the fixed measurements will be written. "
             "if not provided will be set to the measurements directory.",
    )
    parser.add_argument(
        "-k",
        "--keep-orig",
        action="store_true",
        default=False,
        help="whether to keep or remove original measurement files",
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    measurements_path = args.measurements
    output_path = args.out
    same_dir = False
    if output_path == "":
        output_path = measurements_path
        same_dir = True
    elif not os.path.exists(output_path):
        os.makedirs(output_path)
    # If output path is provided we assume it is different than the measurements path.
    remove_orig = not args.keep_orig
    measurements_paths = os.listdir(measurements_path)
    measurements_paths_ranges = [
        measurement_path
        for measurement_path in measurements_paths
        if measurement_path.endswith(".json")
        and "MAXABS_HW" not in measurement_path
        and "mod_list" not in measurement_path
    ]
    measurements_paths_scales = [
        measurement_path
        for measurement_path in measurements_paths
        if measurement_path.endswith(".json")
        and "MAXABS_HW" in measurement_path
        and "mod_list" not in measurement_path
    ]

    for measurement in measurements_paths_ranges + measurements_paths_scales:
        fixed_json_path = os.path.join(output_path, f"{measurement.split(os.sep)[-1]}")
        path = os.path.join(measurements_path, measurement)
        npz_path = path.replace(".json", ".npz")
        if same_dir:
            renamed_path = os.path.join(measurements_path, f"orig_{measurement}")
            renamed_npz_path = renamed_path.replace(".json", ".npz")
            os.rename(path, renamed_path)
            os.rename(npz_path, renamed_npz_path)
            path = renamed_path
            npz_path = renamed_npz_path
        with open(fixed_json_path, "w") as fixed_json_file:
            with open(path, "r") as json_file:
                data_to_fix = json.load(json_file)
                fixed_data = fix_cache_inputs(data_to_fix)

                json.dump(fixed_data, fixed_json_file)

                global_rank = fixed_data["GlobalRank"]
                local_rank = fixed_data["LocalRank"]
                mode = fixed_data["Mode"]
                nodes = fixed_data["Nodes"]
                layers = {}
                fixed_npz_path = fixed_json_path.replace(".json", ".npz")
                for layer, dlayer in nodes.items():
                    layers[layer] = {}
                    layers[layer]["inputs"] = [np.array(x) for x in dlayer["inputs"]]
                    if dlayer.get("outputs") is not None:
                        layers[layer]["outputs"] = [np.array(x) for x in dlayer["outputs"]]
                    if dlayer.get("params") is not None and dlayer["params"].get("weight") is not None:
                        layers[layer]["params"] = {}
                        layers[layer]["params"]["weight"] = np.array(dlayer["params"]["weight"])
                df = {"GlobalRank": global_rank, "LocalRank": local_rank, "Mode": mode, "Nodes": layers}
                with open(fixed_npz_path, "w"):
                    np.savez(fixed_npz_path, df)
        if remove_orig:
            os.remove(path)
            os.remove(npz_path)

    print("Finished postprocessing_measurements script")


if __name__ == "__main__":
    main(sys.argv[1:])
