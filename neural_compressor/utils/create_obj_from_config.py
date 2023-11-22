#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""Utility methods to create corresponding objects from configuration."""

import copy
import gc
from collections import OrderedDict

from neural_compressor.data import DATALOADERS, FILTERS, TRANSFORMS, Datasets
from neural_compressor.metric import METRICS

DEFAULT_BATCH_SIZE = 64


def get_func_from_config(func_dict, cfg, compose=True):
    """Get the function or the composed function from configuration."""
    func_list = []
    for func_name, func_value in OrderedDict(cfg).items():
        func_kwargs = {}
        func_args = []
        if isinstance(func_value, dict):
            func_kwargs = func_value
        elif func_value is not None:
            func_args.append(func_value)
        func_list.append(func_dict[func_name](*func_args, **func_kwargs))

    func = func_dict["Compose"](func_list) if compose else (func_list[0] if len(func_list) > 0 else None)
    return func


def get_preprocess(preprocesses, cfg, compose=True):
    """Get the preprocess function from configuration."""
    return get_func_from_config(preprocesses, cfg, compose)


def get_metrics(metrics, cfg, compose=True):
    """Get the metrics function from configuration."""
    return get_func_from_config(metrics, cfg, compose)


def get_postprocess(postprocesses, cfg, compose=True):
    """Get the postprocess function from configuration."""
    return get_func_from_config(postprocesses, cfg, compose)


def get_algorithm(algorithms, cfg, compose=False):
    """Get the algorithms from configuration.

    Args:
        algorithms: the algorithm management.
        cfg: a dict contain the algo name and use it or not.
        compose: compose all algo or not. Defaults to False.

    Returns:
        All open algos.
    """
    # recipes contains quantization part, only use algorithms in that
    algo_conf = algorithms.support_algorithms().intersection(set(cfg.keys()))
    # (TODO) only support open/close according to cfg
    return [algorithms()[algo] for algo in algo_conf if cfg[algo]]


def create_dataset(framework, data_source, cfg_preprocess, cfg_filter):
    """Create the dataset from the data source."""
    transform_list = []
    # generate framework specific transforms
    preprocess = None
    if cfg_preprocess is not None:
        preprocesses = TRANSFORMS(framework, "preprocess")
        preprocess = get_preprocess(preprocesses, cfg_preprocess)
    # even we can unify transform, how can we handle the IO, or we do the transform here
    datasets = Datasets(framework)
    dataset_type = list(data_source.keys())[0]
    # generate framework and dataset specific filters
    filter = None
    if cfg_filter is not None:
        filters = FILTERS(framework)
        filter_type = list(cfg_filter.keys())[0]
        filter_dataset_type = filter_type + dataset_type
        filter = filters[filter_dataset_type](**cfg_filter[filter_type])
    # in this case we should prepare eval_data and calib_data separately
    dataset = datasets[dataset_type](**data_source[dataset_type], transform=preprocess, filter=filter)
    return dataset


def create_dataloader(framework, dataloader_cfg):
    """Create the dataloader according to the framework."""
    batch_size = (
        int(dataloader_cfg["batch_size"]) if dataloader_cfg.get("batch_size") is not None else DEFAULT_BATCH_SIZE
    )
    last_batch = dataloader_cfg["last_batch"] if dataloader_cfg.get("last_batch") is not None else "rollover"
    shuffle = dataloader_cfg["shuffle"] if dataloader_cfg.get("shuffle") is not None else False
    distributed = dataloader_cfg["distributed"] if dataloader_cfg.get("distributed") is not None else False

    dataset = create_dataset(
        framework,
        copy.deepcopy(dataloader_cfg["dataset"]),
        copy.deepcopy(dataloader_cfg["transform"]),
        copy.deepcopy(dataloader_cfg["filter"]),
    )

    return DATALOADERS[framework](
        dataset=dataset, batch_size=batch_size, last_batch=last_batch, shuffle=shuffle, distributed=distributed
    )


def create_eval_func(
    framework, dataloader, adaptor, metric, postprocess_cfg=None, iteration=-1, tensorboard=False, fp32_baseline=False
):
    """The interface to create evaluate function from config.

    Args:
        framework (str): The string of framework.
        dataloader (common.DataLoader): The object of common.DataLoader.
        adaptor (obj): The object of adaptor.
        metric: The evaluation metric.
        postprocess_cfg: The postprocess configuration.
        iteration: The number of iterations to evaluate.
        tensorboard: Whether to use tensorboard.
        fp32_baseline: The fp32 baseline score.

    Returns:
        The constructed evaluation function
    """
    # eval_func being None means user will provide dataloader and metric info
    # in config yaml file
    assert dataloader, "dataloader should NOT be empty when eval_func is None"
    postprocess = None
    if postprocess_cfg is not None:
        postprocesses = TRANSFORMS(framework, "postprocess")
        postprocess = get_postprocess(postprocesses, postprocess_cfg["transform"])
    if isinstance(metric, dict):
        fwk_metrics = METRICS(framework)
        metrics = []
        for name, val in metric.items():
            if (
                isinstance(val, int)
                and len([i for i in gc.get_objects() if id(i) == val]) > 0
                and "user_" + type([i for i in gc.get_objects() if id(i) == val][0]).__name__ == name
            ):
                metrics.extend([i for i in gc.get_objects() if id(i) == val])
            elif name not in ["weight", "higher_is_better"]:
                metrics.append(get_metrics(fwk_metrics, {name: val}, compose=False))
    else:
        metrics = metric

    def eval_func(model, measurer=None):
        return adaptor.evaluate(
            model, dataloader, postprocess, metrics, measurer, iteration, tensorboard, fp32_baseline
        )

    # TODO: to find a better way
    eval_func.builtin = True

    return eval_func


def create_train_func(framework, dataloader, adaptor, train_cfg, hooks=None, callbacks=None):
    """The interface to create train function from config.

    Args:
        framework (str): The string of framework.
        dataloader (common.DataLoader): The object of common.DataLoader.
        adaptor (obj): The object of adaptor.
        train_cfg (dict): The dict of training related config.
        hooks (dict): The dict of training hooks, supported keys are:
                      on_epoch_begin, on_epoch_end, on_step_begin, on_step_end.
                      Their values are functions to be executed in adaptor layer.

    Returns:
        The constructed train function
    """
    # train_func being None means user will provide training related info
    # in config yaml file
    assert dataloader, "dataloader should NOT be empty when train_func is None"
    assert adaptor, "adaptor should NOT be empty"

    from neural_compressor.compression.distillation.criterions import Criterions
    from neural_compressor.compression.distillation.optimizers import Optimizers

    postprocess_cfg = train_cfg.postprocess
    if postprocess_cfg is not None:
        postprocesses = TRANSFORMS(framework, "postprocess")
        postprocess = get_postprocess(postprocesses, postprocess_cfg["transform"])
    else:
        postprocess = None
    if isinstance(train_cfg.optimizer, dict):
        assert train_cfg.optimizer and len(train_cfg.optimizer) == 1, "optimizer should only set once"
        key, value = next(iter(train_cfg.optimizer.items()))
        optimizer = Optimizers(framework)[key](value)
        optimizer = optimizer()
    else:
        if framework == "pytorch":
            optimizer = (lambda mp, p: train_cfg.optimizer, {"p": 0})
        elif framework == "tensorflow":
            optimizer = (lambda p: train_cfg.optimizer, {"p": 0})

    if isinstance(train_cfg.criterion, dict):
        assert train_cfg.criterion and len(train_cfg.criterion) == 1, "criterion should only set once"
        key, value = next(iter(train_cfg.criterion.items()))
        criterion = Criterions(framework)[next(iter(train_cfg.criterion))](value)
        criterion = criterion()
    else:
        criterion = (lambda p: train_cfg.criterion, {"p": 0})

    default_dict = {k: train_cfg[k] for k in train_cfg.keys() - {"optimizer", "criterion", "dataloader"}}
    default_dict["callbacks"] = callbacks

    def train_func(model):
        return adaptor.train(
            model,
            dataloader,
            optimizer_tuple=optimizer,
            criterion_tuple=criterion,
            hooks=hooks,
            postprocess=postprocess,
            kwargs=default_dict,
        )

    # TODO: to find a better way
    train_func.builtin = True

    return train_func
