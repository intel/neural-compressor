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

import os
from pathlib import Path
import re
import toml
import torch

import numpy as np
from numpy.core.numeric import full
from pytorch_SUT import PytorchSUT

MLPERF_CONF = Path(os.path.dirname(os.path.realpath(__file__))) / "./mlperf.conf"
MLPERF_CONF = MLPERF_CONF.resolve()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server"], default="Offline", help="Scenario")
    # parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--mlperf_conf", default=str(MLPERF_CONF), help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--pytorch_config_toml", default="pytorch/configs/rnnt.toml")
    parser.add_argument("--pytorch_checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--perf_count", type=int, default=None)
    parser.add_argument("--profile", choices=["True", "Split", "False"], default="False")
    parser.add_argument("--bf16", dest='bf16', action='store_true')
    parser.add_argument("--int8", dest='int8', action='store_true')
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--configure_path", default="")
    parser.add_argument('--tune', dest='tune', action='store_true', 
                        help='tune best int8 model with Neural Compressor on calibration dataset')
    parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                        help='run benchmark')
    parser.add_argument("--accuracy_only", dest='accuracy_only', action='store_true',
                        help='For accuracy measurement only.')
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
    print ("Checking args: int8={}, bf16={}".format(args.int8, args.bf16))
    print(args)

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "rnnt", args.scenario)
    settings.FromConfig(args.user_conf, "rnnt", args.scenario)

    if args.accuracy_only:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = args.log_dir
    os.makedirs(log_path, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    pattern = ['accuracy=\d+.\d+', 'samples_per_query : \d+', 'Mean latency.*', 'Samples per second\\s*: \d+.\d+']

    def eval_func(model):
        print("Running Loadgen test...")
        fullpath = None
        use_int8 = False
        settings.mode = lg.TestMode.AccuracyOnly
        for path, dirs, files in os.walk('nc_workspace'):
            if 'ipex_config_tmp.json' in files:
                fullpath = os.path.join(path, 'ipex_config_tmp.json')
                use_int8 = True
                break
        sut = PytorchSUT(args.pytorch_config_toml, args.pytorch_checkpoint,
                         args.dataset_dir, args.manifest, args.perf_count,
                         args.bf16, use_int8, fullpath)
        lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)
        cmd = f"python3 accuracy_eval.py --log_dir {log_path} \
            --dataset_dir {args.dataset_dir} --manifest {args.manifest}"
        out = subprocess.check_output(cmd, shell=True)
        out = out.decode()
        regex_accu = re.compile(pattern[0])
        accu = float(regex_accu.findall(out)[0].split('=')[1])
        print('Accuracy: %.3f ' % (accu))
        return accu

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        import shutil
        shutil.rmtree('nc_workspace', ignore_errors=True)
        sut = PytorchSUT(args.pytorch_config_toml, args.pytorch_checkpoint,
                         args.dataset_dir, args.manifest, args.perf_count,
                         True, False, None)
        model = sut.greedy_decoder._model.encoder

        class NC_dataloader(object):
            def __init__(self, sut):
                self.sut = sut
                self.batch_size = 1

            def __iter__(self):
                for i in range(0, self.sut.qsl.count, self.batch_size):
                    waveform = self.sut.qsl[i]
                    assert waveform.ndim == 1
                    waveform_length = np.array(waveform.shape[0], dtype=np.int64)
                    waveform = np.expand_dims(waveform, 0)
                    waveform_length = np.expand_dims(waveform_length, 0)
                    with torch.no_grad():
                        waveform = torch.from_numpy(waveform)
                        waveform_length = torch.from_numpy(waveform_length)
                        feature, feature_length = self.sut.audio_preprocessor.forward((waveform, waveform_length))
                        assert feature.ndim == 3
                        assert feature_length.ndim == 1
                        feature = feature.permute(2, 0, 1)
                    yield (feature, feature_length), None

        calib_dataloader = NC_dataloader(sut)
        quantizer = Quantization("./conf.yaml")
        quantizer.model = common.Model(model)
        quantizer.calib_dataloader = calib_dataloader
        quantizer.eval_func = eval_func
        q_model = quantizer.fit()
        q_model.save(args.tuned_checkpoint)
        return

    if args.backend == "pytorch":
        config_file = None
        if args.int8:
            config_file = os.path.join(args.tuned_checkpoint, "best_configure.json")
            assert os.path.exists(config_file), "there is no ipex config file, Please tune with Neural Compressor first!"
        sut = PytorchSUT(args.pytorch_config_toml, args.pytorch_checkpoint,
                         args.dataset_dir, args.manifest, args.perf_count,
                         args.bf16, args.int8, config_file)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    if args.accuracy_only:
        cmd = f"python3 accuracy_eval.py --log_dir {log_path} --dataset_dir {args.dataset_dir} --manifest {args.manifest}"
        print(f"Running accuracy script: {cmd}")
        out = subprocess.check_output(cmd, shell=True)
        out = out.decode()
        regex_accu = re.compile(pattern[0])
        accu = float(regex_accu.findall(out)[0].split('=')[1])
        print('Accuracy: %.3f ' % (accu))
    else:
        file_path = os.path.join(log_path, 'mlperf_log_summary.txt')
        f = open(file_path, 'r', encoding='UTF-8')
        file_content = f.read()
        f.close()
        regex_batch = re.compile(pattern[1])
        regex_late = re.compile(pattern[2])
        regex_perf = re.compile(pattern[3], flags=re.IGNORECASE)
        latency_per_sample = float(regex_late.findall(file_content)[0].split(': ')[1])
        samples_per_s = float(regex_perf.findall(file_content)[0].split(': ')[1])
        print('Batch size = %d' % 1)
        print('Latency: %.3f ms' % (latency_per_sample / 10**6))
        print('Throughput: %.3f samples/sec' % (samples_per_s))

    print("Done!")



if __name__ == "__main__":
    main()
