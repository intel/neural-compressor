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

import argparse
import mlperf_loadgen as lg
import subprocess

import time
import os
from pathlib import Path
import re

MLPERF_CONF = Path(os.path.dirname(os.path.realpath(__file__))) / "./mlperf.conf"
MLPERF_CONF = MLPERF_CONF.resolve()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', dest='tune', action='store_true', 
                        help='tune best int8 model on calibration dataset')
    parser.add_argument("--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server"], 
                        default="Offline", help="Scenario")
    parser.add_argument("--mlperf_conf", default=str(MLPERF_CONF), help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf", 
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--pytorch_config_toml", default="pytorch/configs/rnnt.toml")
    parser.add_argument("--pytorch_checkpoint", default="pytorch/work_dir/rnnt.pt")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--perf_count", type=int, default=None)
    parser.add_argument("--log_dir", default='./saved_log')
    parser.add_argument('--performance', dest='performance', action='store_true',
                        help='run benchmark')
    parser.add_argument("--accuracy", dest='accuracy', action='store_true',
                        help='For accuracy measurement only.')
    parser.add_argument('--int8', dest='int8', action='store_true', help='load int8 model')
    parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                        help='path to checkpoint tuned by Neural Compressor (default: ./)')
    args = parser.parse_args()
    return args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = get_args()

    if args.backend == "pytorch":
        from pytorch_SUT import PytorchSUT
        sut = PytorchSUT(args.pytorch_config_toml, args.pytorch_checkpoint,
                        args.dataset_dir, args.manifest, args.perf_count)
        model = sut.greedy_decoder._model
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "rnnt", args.scenario)
    settings.FromConfig(args.user_conf, "rnnt", args.scenario)

    if args.performance:
        settings.mode = lg.TestMode.PerformanceOnly
    else:
        settings.mode = lg.TestMode.AccuracyOnly

    log_path = args.log_dir
    os.makedirs(log_path, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    pattern = ['accuracy=\d+.\d+', 'samples_per_query : \d+', 'Mean latency.*']
    
    def eval_func(model):
        print("Running Loadgen test...")
        sut.greedy_decoder._model = model
        lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)
        cmd = f"python3 accuracy_eval.py --log_dir {log_path} \
            --dataset_dir {args.dataset_dir} --manifest {args.manifest}"
        out = subprocess.check_output(cmd, shell=True)
        out = out.decode()
        regex_accu = re.compile(pattern[0])
        accu = float(regex_accu.findall(out)[0].split('=')[1])
        print('Accuracy: %.3f ' % (accu))
        return accu
    
    def benchmark(model):
        print("Running Loadgen test...")
        sut.greedy_decoder._model = model
        lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)
        file_path = os.path.join(log_path, 'mlperf_log_summary.txt')
        f = open(file_path, 'r', encoding='UTF-8')
        file_content = f.read()
        f.close()
        regex_batch = re.compile(pattern[1])
        regex_late = re.compile(pattern[2])
        samples_per_query = int(regex_batch.findall(file_content)[0].split(': ')[1])
        latency_per_sample = int(regex_late.findall(file_content)[0].split(': ')[1])
        print('Batch size = %d' % samples_per_query)
        print('Latency: %.3f ms' % (latency_per_sample / 10**6))
        print('Throughput: %.3f samples/sec' % (10**9/latency_per_sample))

    if args.tune:
        from neural_compressor import PostTrainingQuantConfig
        from neural_compressor import quantization
        conf = PostTrainingQuantConfig(approach="dynamic")
        q_model = quantization.fit(model,
                                    conf,
                                    eval_func=eval_func)
        q_model.save(args.tuned_checkpoint)
        return

    elif args.int8:
        from neural_compressor.utils.pytorch import load
        int8_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)), model)
        if args.accuracy:
            eval_func(int8_model)
        elif args.performance:
            benchmark(int8_model)
    else:
        if args.accuracy:
            eval_func(model)
        elif args.performance:
            benchmark(model)
        

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
