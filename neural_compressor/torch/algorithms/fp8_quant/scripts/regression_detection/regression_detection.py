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

import numpy as np
import scipy

tasks = ["winogrande", "hellaswag", "piqa", "lambada_openai"]


def ztest(ref_mean=0.0, ref_stderr=1.0, test_mean=0.0, test_stderr=0.0):
    z_score = (test_mean - ref_mean) / np.sqrt(ref_stderr**2 + test_stderr**2)
    p_value = 1.0 + scipy.special.erf(-np.abs(z_score) / np.sqrt(2))
    return p_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Regression detection using Z-Test. We assume we have mean and SEM of golden run arranged in json and test results json and we compare the results to see if degregation occurred.",
    )
    parser.add_argument(
        "--hp_dtype",
        type=str,
        help="Data type of the high precision test",
        default=None,
    )
    parser.add_argument(
        "--lp_dtype",
        type=str,
        help="Data type of the low precision test",
        default=None,
    )
    parser.add_argument(
        "--golden_metrics",
        type=str,
        help="Path to json that includes mean, SEM and diff golden metrics of bf16 and fp8 precision.",
        default=None,
    )
    parser.add_argument(
        "--test_metrics_lp",
        type=str,
        help="Path to json that includes mean, SEM and diff test metrics of lp precision.",
        default=None,
    )
    parser.add_argument(
        "--test_metrics_hp",
        type=str,
        help="Path to json that includes mean, SEM and diff test metrics of high precision.",
        default=None,
    )
    parser.add_argument("--quantization_mode", type=str, help="quantization mode", default=None)
    args = parser.parse_args()
    mode = args.quantization_mode
    hp_dtype = args.hp_dtype
    lp_dtype = args.lp_dtype
    golden_metrics_path = args.golden_metrics
    test_metrics_lp_path = args.test_metrics_lp
    test_metrics_hp_path = args.test_metrics_hp
    if golden_metrics_path is None or test_metrics_hp_path is None or test_metrics_lp_path is None:
        print("Please provide golden_metrics, test_metrics_hp_path and test_metrics_lp_path json paths")
        exit(1)

    with open(golden_metrics_path, "r") as f:
        golden_metrics_json = json.load(f)

    with open(test_metrics_lp_path, "r") as f:
        test_metrics_lp_json = json.load(f)
        test_metrics_lp_json = test_metrics_lp_json["results"]

    with open(test_metrics_hp_path, "r") as f:
        test_metrics_hp_json = json.load(f)
        test_metrics_hp_json = test_metrics_hp_json["results"]

    regressions = []
    for task in tasks:
        # The two-sample z-test comparing the golden and under-test high-precision configuration
        ref_mean_hp = golden_metrics_json[hp_dtype][task]["mean"]
        ref_stderr_hp = golden_metrics_json[hp_dtype][task]["sem"]
        test_mean_hp = test_metrics_hp_json[task]["acc"]
        test_stderr_hp = test_metrics_hp_json[task]["acc_stderr"]
        p_hp_value = ztest(ref_mean_hp, ref_stderr_hp, test_mean_hp, test_stderr_hp)
        print(f"Z-Test high precision p-value={p_hp_value*100:.2f}%  in {task} task")
        if p_hp_value < 0.05:
            regressions.append(f"Z-Test high precision p-value is less than 0.05 in {task} task.")

        # The two-sample z-test comparing the golden and under-test low-precision configuration
        if mode is not None:
            ref_mean_lp = golden_metrics_json[lp_dtype][mode][task]["mean"]
            ref_stderr_lp = golden_metrics_json[lp_dtype][mode][task]["sem"]
        else:
            ref_mean_lp = golden_metrics_json[lp_dtype][task]["mean"]
            ref_stderr_lp = golden_metrics_json[lp_dtype][task]["sem"]
        test_mean_lp = test_metrics_lp_json[task]["acc"]
        test_stderr_lp = test_metrics_lp_json[task]["acc_stderr"]
        p_lp_value = ztest(ref_mean_lp, ref_stderr_lp, test_mean_lp, test_stderr_lp)
        print(f"Z-Test low precision p-value={p_lp_value*100:.2f}% in {task} task")
        if p_lp_value < 0.05:
            regressions.append(f"Z-Test low precision p-value is less than 0.05 in {task} task.")

        # The single-sample z-test comparing the golden and under-test degradation of low-precision configuration
        if mode is not None:
            ref_mean_diff = golden_metrics_json[lp_dtype][mode][task]["mean_diff"]
            ref_stderr_diff = golden_metrics_json[lp_dtype][mode][task]["sem_diff"]
        else:
            ref_mean_diff = golden_metrics_json[lp_dtype][task]["mean_diff"]
            ref_stderr_diff = golden_metrics_json[lp_dtype][task]["sem_diff"]
        test_mean_diff = test_mean_lp - test_mean_hp
        p_diff_value = ztest(ref_mean_diff, ref_stderr_diff, test_mean_diff, ref_stderr_diff)
        print(f"Z-Test low precision diff p-value={p_diff_value*100:.2f}% in {task} task")
        if p_diff_value < 0.05:
            regressions.append(f"Z-Test low precision diff p-value is less than 0.05 in {task} task.")

    if len(regressions) == 0:
        print("No regressions were detected!")
    else:
        print("Regressions were detected!")
        for regression in regressions:
            print(regression)
