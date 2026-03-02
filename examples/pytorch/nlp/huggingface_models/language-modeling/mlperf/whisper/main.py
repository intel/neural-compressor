# Copyright 2020 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import argparse
import subprocess
from pathlib import Path

import mlperf_loadgen as lg
from sut import vllmSUT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model_path", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--audit_conf", default="audit.config", help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--perf_count", type=int, default=None)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--num_workers", default=1, type=int)
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = get_args()
    print(args)

    log_path = args.log_dir
    os.makedirs(log_path, exist_ok=True)


    sut = vllmSUT(dataset_dir=args.dataset_dir, 
                  model_path=args.model_path,
                  manifest_filepath=args.manifest, 
                  perf_count=args.perf_count, 
                  num_workers=args.num_workers)
    sut.start()

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.user_conf, "whisper", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, 
                                sut.qsl.qsl, 
                                settings, 
                                log_settings, 
                                args.audit_conf)
    sut.stop()

    if args.accuracy:
        acc_path = open(log_path + "/accuracy.txt", "w")
        cmd = ["python3", "accuracy.py", "--log_dir", log_path, "--dataset_dir", args.dataset_dir, "--manifest", args.manifest]
        subprocess.check_call(cmd, stdout=acc_path)

    print("Done!")


if __name__ == "__main__":
    main()