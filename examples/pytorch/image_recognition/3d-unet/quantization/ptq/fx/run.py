# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020 - 2021 INTEL CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

import os
import sys
sys.path.insert(0, os.getcwd())
import time

import argparse
import mlperf_loadgen as lg
import subprocess
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


parser = argparse.ArgumentParser(description='PyTorch 3d-unet Training')
parser = argparse.ArgumentParser()
parser.add_argument("--backend",
                    choices=["pytorch", "onnxruntime", "tf", "ov"],
                    default="pytorch",
                    help="Backend")
parser.add_argument(
    "--scenario",
    choices=["SingleStream", "Offline", "Server", "MultiStream"],
    default="Offline",
    help="Scenario")
parser.add_argument("--accuracy",
                    action="store_true",
                    help="enable accuracy pass")
parser.add_argument("--mlperf_conf",
                    default="build/mlperf.conf",
                    help="mlperf rules config")
parser.add_argument("--user_conf",
                    default="user.conf",
                    help="user config for user LoadGen settings such as target QPS")
parser.add_argument(
    "--model_dir",
    default=
    "build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1",
    help="Path to the directory containing plans.pkl")
parser.add_argument("--model", help="Path to the ONNX, OpenVINO, or TF model")
parser.add_argument("--preprocessed_data_dir",
                    default="build/preprocessed_data",
                    help="path to preprocessed data")
parser.add_argument("--performance_count",
                    type=int,
                    default=16,
                    help="performance count")
parser.add_argument('--tune', dest='tune', action='store_true',
                    help='tune best int8 model on calibration dataset')
parser.add_argument('--performance', dest='performance', action='store_true',
                    help='run benchmark')
parser.add_argument('--int8', dest='int8', action='store_true',
                    help='run benchmark for int8')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('-i', "--iter", default=0, type=int,
                    help='For accuracy measurement only.')
parser.add_argument('-w', "--warmup_iter", default=5, type=int,
                    help='For benchmark measurement only.')
parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                    help='path to checkpoint tuned by Neural Compressor'
                         ' (default: ./)')
args = parser.parse_args()


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}


def eval_func(model):
    if args.backend == "pytorch":
        from pytorch_SUT import get_pytorch_sut
        sut = get_pytorch_sut(model, args.preprocessed_data_dir,
                              args.performance_count)
    elif args.backend == "onnxruntime":
        from onnxruntime_SUT import get_onnxruntime_sut
        sut = get_onnxruntime_sut(args.model, args.preprocessed_data_dir,
                                  args.performance_count)
    elif args.backend == "tf":
        from tf_SUT import get_tf_sut
        sut = get_tf_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    elif args.backend == "ov":
        from ov_SUT import get_ov_sut
        sut = get_ov_sut(args.model, args.preprocessed_data_dir,
                         args.performance_count)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario)
    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly

    log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    if args.accuracy:
        print("Running accuracy script...")
        process = subprocess.Popen(['python3', 'accuracy-brats.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()

        print(out)
        print("Batch size = ", args.batch_size)
        print("Accuracy: ", float(err))

    lg.DestroySUT(sut.sut)
    lg.DestroyQSL(sut.qsl.qsl)
    return float(err)


sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))
from nnunet.training.model_restore import load_model_and_checkpoint_files
from neural_compressor.experimental import Quantization, common
import pickle


def main():
    args = parser.parse_args()

    class CalibrationDL():
        def __init__(self):
            path = os.path.abspath(os.path.expanduser('./brats_cal_images_list.txt'))
            with open(path, 'r') as f:
                self.preprocess_files = [line.rstrip() for line in f]
                
            self.loaded_files = {}
            self.batch_size = 1

        def __getitem__(self, sample_id):
            file_name = self.preprocess_files[sample_id]
            print("Loading file {:}".format(file_name))
            with open(os.path.join('build/calib_preprocess/', "{:}.pkl".format(file_name)), "rb") as f:
                self.loaded_files[sample_id] = pickle.load(f)[0]
            # note that calibration phase does not care label, here we return 0 for label free case.
            return self.loaded_files[sample_id], 0

        def __len__(self):
            self.count = len(self.preprocess_files)
            return self.count
        

    assert args.backend == "pytorch"
    model_path = os.path.join(args.model_dir, "plans.pkl")
    assert os.path.isfile(model_path), "Cannot find the model file {:}!".format(model_path)
    trainer, params = load_model_and_checkpoint_files(args.model_dir, folds=1, fp16=False
                                                      , checkpoint_name='model_final_checkpoint')
    trainer.load_checkpoint_ram(params[0], False)
    model = trainer.network
    
    from neural_compressor.data import dataloaders
    dataloader = dataloaders.DataLoader(dataset=CalibrationDL(), framework="pytorch_fx")

    if args.tune:
        from neural_compressor import PostTrainingQuantConfig
        from neural_compressor import quantization
        conf = PostTrainingQuantConfig()
        q_model = quantization.fit(model,
                                    conf,
                                    calib_dataloader=dataloader,
                                    eval_func=eval_func)
        q_model.save(args.tuned_checkpoint)
        return

    if args.performance or args.accuracy:
        model.eval()
        if args.int8:
            from neural_compressor.utils.pytorch import load
            new_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)),
                             model,
                             dataloader=dataloader)
        else:
            new_model = model
        if args.performance:
            from neural_compressor.config import BenchmarkConfig
            from neural_compressor import benchmark
            b_conf = BenchmarkConfig(warmup=5,
                                     iteration=args.iter,
                                     cores_per_instance=4,
                                     num_of_instance=1)
            benchmark.fit(new_model, b_conf, b_dataloader=dataloader)
        if args.accuracy:
            eval_func(new_model)
        return


if __name__ == "__main__":
    main()
