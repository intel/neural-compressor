"""Simulated Annealing Optimizer."""

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

import math
import time
from random import random

import numpy as np

try:
    from neural_compressor.utils import logger
except:
    import logging

    logger = logging.getLogger("sa_optimizer")


class SimulatedAnnealingOptimizer(object):
    def __init__(
        self,
        generate_func=None,
        T0=100,
        Tf=0.01,
        higher_is_better=True,
        alpha=None,
        iter=500,
        early_stop=50,
        log_interval=50,
    ):
        """Initialize."""
        self.generate_func = generate_func
        self.T0 = T0
        self.Tf = Tf
        self.T = self.T0
        self.higher_is_better = higher_is_better
        self.alpha = alpha
        self.iter = iter
        self.early_stop = early_stop
        self.log_interval = log_interval
        self.best = (float("-inf"), None) if self.higher_is_better else (float("inf"), None)
        self.history = {"T": [], "F": []}

    def _metrospolis(self, f, f_new):
        if (not self.higher_is_better and f_new <= f) or (self.higher_is_better and f_new >= f):
            return 1
        else:
            if self.higher_is_better:
                p = math.exp((f_new - f) / self.T)
            else:
                p = math.exp((f - f_new) / self.T)
            if random() < p:
                return 1
            else:
                return 0

    def _generate_new_points(self, points):
        new_points = np.array(points)
        new_points += self.T * (np.random.random(new_points.shape) - np.random.random(new_points.shape))
        return new_points

    def gen_next_params(self, func, points):
        """Get the next parameter."""
        count = 0
        last_modify = 0
        self.T = self.T0
        self.best = (float("-inf"), None) if self.higher_is_better else (float("inf"), None)
        scores = func(points)

        self.history = {"T": [], "F": [], "P": []}
        st = time.time()

        while self.T > self.Tf:
            # generate new points
            if self.generate_func:
                new_points = self.generate_func(points)
            else:
                new_points = self._generate_new_points(points)
            new_scores = func(new_points)
            for i, s in enumerate(new_scores):
                if self._metrospolis(scores[i], s):
                    points[i] = new_points[i]
                    scores[i] = s
                if (not self.higher_is_better and scores[i] < self.best[0]) or (
                    self.higher_is_better and scores[i] > self.best[0]
                ):
                    last_modify = count
                    self.best = (scores[i], [float(v) for v in points[i]])

            self.history["T"].append(self.T)
            if self.higher_is_better:
                self.history["F"].append(max(scores))
                self.history["P"].append(points[np.argmax(scores)])
            else:
                self.history["F"].append(min(scores))
                self.history["P"].append(points[np.argmax(scores)])

            if self.alpha:
                self.T *= self.alpha
            else:
                self.T -= (self.T0 - self.Tf) / (self.iter + 1)
            count += 1

            if self.log_interval and count % self.log_interval == 0:
                elapse = time.time() - st
                logger.debug(
                    f"SA iter: {count}\tlast_update: {last_modify}\t \
                             max score: {self.best[0]}\tpoint: {self.best[1]}\t \
                             temp: {self.T}\telasped: {elapse}"
                )

            if count - last_modify > self.early_stop:
                break
        return self.best[1]
