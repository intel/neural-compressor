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

from lpot.experimental.metric import METRICS
from lpot.experimental.data import DATASETS, TRANSFORMS, FILTERS, DATALOADERS
from collections import OrderedDict
import copy

def get_func_from_config(func_dict, cfg, compose=True):
    func_list = []
    for func_name, func_value in OrderedDict(cfg).items():
        func_kwargs = {}
        func_args = []
        if isinstance(func_value, dict):
            func_kwargs = func_value
        elif func_value is not None:
            func_args.append(func_value)
        func_list.append(func_dict[func_name](*func_args, **func_kwargs))

    func = func_dict['Compose'](func_list) if compose else \
        (func_list[0] if len(func_list) > 0 else None)
    return func


def get_preprocess(preprocesses, cfg, compose=True):
    return get_func_from_config(preprocesses, cfg, compose)


def get_metrics(metrics, cfg, compose=True):
    return get_func_from_config(metrics, cfg, compose)


def get_postprocess(postprocesses, cfg, compose=True):
    return get_func_from_config(postprocesses, cfg, compose)

def create_dataset(framework, data_source, cfg_preprocess, cfg_filter):
    transform_list = []
    # generate framework specific transforms
    preprocess = None
    if cfg_preprocess is not None:
        preprocesses = TRANSFORMS(framework, 'preprocess')
        preprocess = get_preprocess(preprocesses, cfg_preprocess)
    # even we can unify transform, how can we handle the IO, or we do the transform here
    datasets = DATASETS(framework)
    dataset_type = list(data_source.keys())[0]
    # generate framework and dataset specific filters
    filter = None
    if cfg_filter is not None:
        filters = FILTERS(framework)
        filter_type = list(cfg_filter.keys())[0]
        filter_dataset_type = filter_type + dataset_type
        filter = filters[filter_dataset_type](list(cfg_filter[filter_type].values())[0])
    # in this case we should prepare eval_data and calib_data sperately
    dataset = datasets[dataset_type](**data_source[dataset_type], \
            transform=preprocess, filter=filter)
    return dataset

def create_dataloader(framework, dataloader_cfg):

    batch_size = int(dataloader_cfg['batch_size']) \
        if dataloader_cfg.get('batch_size') is not None else 1
    last_batch = dataloader_cfg['last_batch'] \
        if dataloader_cfg.get('last_batch') is not None else 'rollover'

    eval_dataset = create_dataset(framework,
                                  copy.deepcopy(dataloader_cfg['dataset']),
                                  copy.deepcopy(dataloader_cfg['transform']),
                                  copy.deepcopy(dataloader_cfg['filter']),)

    return DATALOADERS[framework](dataset=eval_dataset, batch_size=batch_size, \
        last_batch=last_batch)

def create_eval_func(framework, dataloader, adaptor, \
                     metric_cfg, postprocess_cfg=None, \
                     iteration=-1, tensorboard=False,
                     fp32_baseline=False):
    """The interface to create evaluate function from config.

    Args:
        model (object): The model to be evaluated.

    Returns:
        Objective: The objective value evaluated
    """

    # eval_func being None means user will provide dataloader and metric info
    # in config yaml file
    assert dataloader, "dataloader should NOT be empty when eval_func is None"
    postprocess = None
    if postprocess_cfg is not None:
        postprocesses = TRANSFORMS(framework, "postprocess")
        postprocess = get_postprocess(postprocesses, postprocess_cfg['transform'])
    if metric_cfg is not None:
        assert len(metric_cfg) == 1, "Only one metric should be specified!"
        metrics = METRICS(framework)
        # if not do compose will only return the first metric
        metric = get_metrics(metrics, metric_cfg, compose=False)
    else:
        metric = None
    def eval_func(model, measurer=None):
        return adaptor.evaluate(model, dataloader, postprocess, \
                                metric, measurer, iteration, \
                                tensorboard, fp32_baseline)

    return eval_func

