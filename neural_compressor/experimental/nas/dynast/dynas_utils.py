#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import csv
import time
import uuid
from datetime import datetime
from typing import Tuple

import numpy as np
import ofa
from fvcore.nn import FlopCountAnalysis
from ofa.imagenet_classification.data_providers.imagenet import ImagenetDataProvider
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager
from ofa.tutorial.flops_table import rm_bn_from_net

from neural_compressor.experimental.nas.dynast.dynas_manager import ParameterManager
from neural_compressor.experimental.nas.dynast.dynas_predictor import Predictor
from neural_compressor.utils.utility import LazyImport, logger

torch = LazyImport('torch')
torchvision = LazyImport('torchvision')


def get_macs(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu',
) -> int:
    model = model.to(device)
    rm_bn_from_net(model)
    model.eval()
    inputs = torch.randn(*input_size, device=device)
    macs = FlopCountAnalysis(model, inputs).total() // input_size[0]
    return macs


def _auto_steps(
    batch_size: int,
    is_warmup: bool = False,
    warmup_scale: float = 5.0,
    min_steps: int = 25,
    min_samples: int = 500,
) -> int:
    """Simple scaling of number of steps w.r.t batch_size

    Example:
    1. `_auto_steps(1, True), _auto_steps(1, False)` -> 100, 500
    2. `_auto_steps(8, True), _auto_steps(8, False)` -> 12, 62
    3. `_auto_steps(16, True), _auto_steps(8, False)` -> 6, 31
    4. `_auto_steps(32, True), _auto_steps(8, False)` -> 5, 25

    Args:
    - batch_size
    - is_warmup: if set to True, will scale down the number of steps by `warmup_scale`.
    - warmup_scale: scale by which number of steps should be decreased if `is_warmup` is True.
    - min_steps: minimum number of steps to return
    - min_samples: returned steps multiplied by `batch_size` should be at least this much.

    Returns:
        number of steps
    """

    if not is_warmup:
        warmup_scale = 1.0

    return int(max(batch_size*min_steps, min_samples)//batch_size//warmup_scale)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    warmup_steps: int = None,
    measure_steps: int = None,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """Measure Torch model's latency.

    Args:
    - model: Torch model
    - input_size: a tuple (batch size, channels, resolution, resolution).
    - warmup_steps - how many data batches to use to warm up the device.
        If 'None' it will be adjusted automatically w.r.t batch size.
    - measure_steps - how many data batches to use for latency measurement.
        If 'None' it will be adjusted automatically w.r.t batch size.
    - device: which device is being used for latency measurement.

    Returns:
        mean latency; std latency
    """
    if not warmup_steps:
        warmup_steps = _auto_steps(input_size[0], is_warmup=True)
    if not measure_steps:
        measure_steps = _auto_steps(input_size[0])

    times = []

    inputs = torch.randn(input_size, device=device)
    model = model.eval()
    rm_bn_from_net(model)
    model = model.to(device)

    if 'cuda' in str(device):
        torch.cuda.synchronize()
    for _ in range(warmup_steps):
        model(inputs)
    if 'cuda' in str(device):
        torch.cuda.synchronize()

    for _ in range(measure_steps):
        if 'cuda' in str(device):
            torch.cuda.synchronize()
        st = time.time()
        model(inputs)
        if 'cuda' in str(device):
            torch.cuda.synchronize()
        ed = time.time()
        times.append(ed - st)

    # Convert to s->ms, round to 0.001
    latency_mean = np.round(np.mean(times) * 1e3, 3)
    latency_std = np.round(np.std(times) * 1e3, 3)

    return latency_mean, latency_std


class Runner:
    pass


class OFARunner(Runner):
    """The OFARunner class manages the sub-network selection from the OFA super-network and
    the validation measurements of the sub-networks. ResNet50, MobileNetV3 w1.0, and MobileNetV3 w1.2
    are currently supported. Imagenet is required for these super-networks `imagenet-ilsvrc2012`.
    """

    def __init__(
        self,
        supernet: str,
        acc_predictor: Predictor,
        macs_predictor: Predictor,
        latency_predictor: Predictor,
        imagenetpath: str,
        batch_size: int,
    ) -> None:
        self.supernet = supernet
        self.acc_predictor = acc_predictor
        self.macs_predictor = macs_predictor
        self.latency_predictor = latency_predictor
        self.device = 'cpu'
        self.test_size = None
        ImagenetDataProvider.DEFAULT_PATH = imagenetpath
        self.ofa_network = ofa.model_zoo.ofa_net(supernet, pretrained=True)
        self.run_config = ImagenetRunConfig(test_batch_size=64, n_worker=20)
        self.batch_size = batch_size

    def estimate_accuracy_top1(
        self,
        subnet_cfg: dict,
    ) -> float:
        top1 = self.acc_predictor.predict(subnet_cfg)
        return top1

    def estimate_macs(
        self,
        subnet_cfg: dict,
    ) -> int:
        macs = self.macs_predictor.predict(subnet_cfg)
        return macs

    def estimate_latency(
        self,
        subnet_cfg: dict,
    ) -> float:
        latency = self.latency_predictor.predict(subnet_cfg)
        return latency

    def validate_top1(
        self,
        subnet_cfg: dict,
    ) -> float: # pragma: no cover
        subnet = self.get_subnet(subnet_cfg)
        folder_name = '.torch/tmp-{}'.format(uuid.uuid1().hex)
        run_manager = RunManager(
            '{}/eval_subnet'.format(folder_name), subnet, self.run_config, init=False
        )
        run_manager.reset_running_statistics(net=subnet)

        # Test sampled subnet
        self.run_config.data_provider.assign_active_img_size(subnet_cfg['r'][0])
        loss, acc = run_manager.validate(net=subnet, no_logs=False)
        top1 = acc[0]
        return top1

    def validate_macs(
        self,
        subnet_cfg: dict,
    ) -> float:
        """Measure Torch model's FLOPs/MACs as per FVCore calculation
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            `macs`
        """

        model = self.get_subnet(subnet_cfg)
        input_size = (self.batch_size, 3, 224, 224)
        macs = get_macs(model=model, input_size=input_size, device=self.device)
        logger.info('Model\'s macs: {}'.format(macs))
        return macs

    @torch.no_grad()
    def measure_latency(
        self,
        subnet_cfg: dict,
        warmup_steps: int = None,
        measure_steps: int = None,
    ) -> Tuple[float, float]:
        """Measure OFA model's latency.
        Args:
            subnet_cfg: sub-network Torch model
        Returns:
            mean latency; std latency
        """
        model = self.get_subnet(subnet_cfg)
        input_size = (self.batch_size, 3, 224, 224)

        latency_mean, latency_std = measure_latency(
            model=model,
            input_size=input_size,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            device=self.device,
        )
        logger.info('Model\'s latency: {} +/- {}'.format(latency_mean, latency_std))

        return latency_mean, latency_std

    def get_subnet(
        self,
        subnet_cfg: dict,
    ) -> torch.nn.Module:
        if self.supernet == 'ofa_resnet50':
            self.ofa_network.set_active_subnet(
                ks=subnet_cfg['d'], e=subnet_cfg['e'], d=subnet_cfg['w']
            )
        else:
            self.ofa_network.set_active_subnet(
                ks=subnet_cfg['ks'], e=subnet_cfg['e'], d=subnet_cfg['d']
            )

        self.subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
        self.subnet.eval()
        return self.subnet


class EvaluationInterface:
    """
    The interface class update is required to be updated for each unique SuperNetwork
    framework as it controls how evaluation calls are made from DyNAS-T

    Args:
        evaluator : class
            The 'runner' that performs the validation or prediction
        manager : class
            The DyNAS-T manager that translates between PyMoo and the parameter dict
        csv_path : string
            (Optional) The csv file that get written to during the subnetwork search
    """

    def __init__(
        self,
        evaluator: Runner,
        manager: ParameterManager,
        metrics: list = ['acc', 'macs'],
        predictor_mode: bool = False,
        csv_path: str = None,
    ) -> None:
        self.evaluator = evaluator
        self.manager = manager
        self.metrics = metrics
        self.predictor_mode = predictor_mode
        self.csv_path = csv_path

    def eval_subnet(
        self,
        x: list,
    ) -> Tuple[dict, float, float]:
        pass

    def clear_csv(self) -> None:
        if self.csv_path:
            f = open(self.csv_path, "w")
            writer = csv.writer(f)
            result = ['Sub-network', 'Date', 'Latency (ms)', ' MACs', 'Top-1 Acc (%)']
            writer.writerow(result)
            f.close()


class EvaluationInterfaceResNet50(EvaluationInterface):
    def __init__(
        self,
        evaluator: Runner,
        manager: ParameterManager,
        metrics: list = ['acc', 'macs'],
        predictor_mode: bool = False,
        csv_path: str = None,
    ) -> None:
        super().__init__(evaluator, manager, metrics, predictor_mode, csv_path)

    def eval_subnet(
        self,
        x: list,
    ) -> Tuple[dict, float, float]:
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        sample = {
            'd': param_dict['d'],
            'e': param_dict['e'],
            'w': param_dict['w'],
            'r': [224],
        }
        subnet_sample = copy.deepcopy(sample)

        # Always evaluate/predict top1
        lat, macs = 0, 0
        if self.predictor_mode == True:
            top1 = self.evaluator.estimate_accuracy_top1(
                self.manager.onehot_generic(x).reshape(1, -1)
            )[0]
            if 'macs' in self.metrics:
                macs = self.evaluator.estimate_macs(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
            if 'lat' in self.metrics:
                lat = self.evaluator.estimate_latency(
                    self.manager.onehot_generic(x).reshape(1, -1)
                )[0]
        else:
            top1 = self.evaluator.validate_top1(subnet_sample)
            macs = self.evaluator.validate_macs(subnet_sample)
            if 'lat' in self.metrics:
                lat, _ = self.evaluator.measure_latency(subnet_sample)

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, lat, macs, top1]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        if 'lat' in self.metrics:
            return sample, lat, -top1
        else:
            return sample, macs, -top1


class EvaluationInterfaceMobileNetV3(EvaluationInterface):
    def __init__(
        self,
        evaluator: Runner,
        manager: ParameterManager,
        metrics=['acc', 'macs'],
        predictor_mode=False,
        csv_path=None,
    ) -> None:
        super().__init__(evaluator, manager, metrics, predictor_mode, csv_path)

    def eval_subnet(
        self,
        x: list,
    ) -> Tuple[dict, float, float]:
        # PyMoo vector to Elastic Parameter Mapping
        param_dict = self.manager.translate2param(x)

        sample = {
            'wid': None,
            'ks': param_dict['ks'],
            'e': param_dict['e'],
            'd': param_dict['d'],
            'r': [224],
        }
        subnet_sample = copy.deepcopy(sample)

        # Always evaluate/predict top1
        lat, macs = 0, 0
        if self.predictor_mode == True:
            top1 = self.evaluator.estimate_accuracy_top1(self.manager.onehot_generic(x).reshape(1,-1))[0]
            if 'macs' in self.metrics:
                macs = self.evaluator.estimate_macs(self.manager.onehot_generic(x).reshape(1,-1))[0]
            if 'lat' in self.metrics:
                lat = self.evaluator.estimate_latency(self.manager.onehot_generic(x).reshape(1,-1))[0]
        else:
            top1 = self.evaluator.validate_top1(subnet_sample)
            macs = self.evaluator.validate_macs(subnet_sample)
            if 'lat' in self.metrics:
                lat, _ = self.evaluator.measure_latency(subnet_sample)

        if self.csv_path:
            with open(self.csv_path, 'a') as f:
                writer = csv.writer(f)
                date = str(datetime.now())
                result = [subnet_sample, date, lat, macs, top1]
                writer.writerow(result)

        # PyMoo only minimizes objectives, thus accuracy needs to be negative
        # Requires format: subnetwork, objective x, objective y
        if 'lat' in self.metrics:
            return sample, lat, -top1
        else:
            return sample, macs, -top1


def get_torchvision_model(
    model_name: str,
) -> torch.nn.Module:
    try:
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model.eval()
        return model
    except AttributeError as ae: # pragma: no cover
        logger.error(
            'Model {model_name} not available. This can be due to either a typo or the model is not '
            'available in torchvision=={torchvision_version}. \nAvailable models: {available_models}'.format(
                model_name=model_name,
                torchvision_version=torchvision.__version__,
                available_models=', '.join(
                    [m for m in dir(torchvision.models) if not m.startswith('_')]
                ),
            )
        )
        raise ae


class TorchVisionReference:
    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        batch_size: int,
        input_size: int = 224,
    ) -> None:
        if 'ofa_resnet50' in model_name:
            model_name = 'resnet50'
        if 'ofa_mbv3' in model_name:
            model_name = 'mobilenet_v3_large'

        self.model_name = model_name
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.input_size = input_size

        logger.info(
            '{name} for \'{model_name}\' on \'{val_dataset_path}\' dataset'.format(
                name=str(self),
                model_name=self.model_name,
                val_dataset_path=self.dataset_path,
            )
        )

        # Here we only download the model. Model will have to be loaded for each validation and benchmarking call
        # separately to avoid modifications to the model being passed between calls.
        get_torchvision_model(model_name=self.model_name)

    def validate_top1(self) -> Tuple[float, float, float]: # pragma: no cover
        ImagenetDataProvider.DEFAULT_PATH = self.dataset_path
        model = get_torchvision_model(model_name=self.model_name)
        run_config = ImagenetRunConfig(test_batch_size=64, n_worker=20)
        folder_name = '.torch/tmp-{}'.format(uuid.uuid1().hex)
        run_manager = RunManager(
            '{}/eval_subnet'.format(folder_name), model, run_config, init=False
        )
        run_config.data_provider.assign_active_img_size(224)
        loss, acc = run_manager.validate(net=model, no_logs=False)
        top1, top5 = acc
        return loss, top1, top5

    def validate_macs(
        self,
        device: str = 'cpu',
    ) -> int:
        model = get_torchvision_model(model_name=self.model_name)
        input_size = (self.batch_size, 3, self.input_size, self.input_size)

        macs = get_macs(model=model, input_size=input_size, device=device)
        logger.info(
            '\'{model_name}\' macs {macs}'.format(
                model_name=self.model_name,
                macs=macs,
            )
        )
        return macs

    def measure_latency(
        self,
        device: str = 'cpu',
        warmup_steps: int = None,
        measure_steps: int = None,
    ) -> Tuple[float, float]:
        model = get_torchvision_model(model_name=self.model_name)
        input_size = (self.batch_size, 3, self.input_size, self.input_size)

        latency_mean, latency_std = measure_latency(
            model=model,
            input_size=input_size,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            device=device,
        )
        logger.info(
            '\'{model_name}\' mean latency {latency_mean} +/- {latency_std}'.format(
                model_name=self.model_name,
                latency_mean=latency_mean,
                latency_std=latency_std,
            )
        )
        return latency_mean, latency_std
