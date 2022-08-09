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

import time

import autograd.numpy as anp
import numpy as np
import pymoo
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize

from neural_compressor.experimental.nas.dynast.dynas_utils import EvaluationInterface
from neural_compressor.utils import logger


class SearchAlgoManager:
    """
    Manages the search parameters for the DyNAS-T single/multi-objective search.

    Args:
        algorithm : string
            Define a multi-objective search algorithm.
        seed : int
            Seed value for pymoo search.
        verbose : Boolean
            Verbosity option
        engine : string
            Support different engine types (e.g. pymoo, optuna, etc.)
    """

    def __init__(
        self,
        algorithm: str = 'nsga2',
        seed: int = 0,
        verbose: bool = False,
        engine: str = 'pymoo',
    ) -> None:
        self.algorithm = algorithm
        self.seed = seed
        self.verbose = verbose
        self.engine = engine
        if self.algorithm == 'nsga2':
            self.configure_nsga2()
            self.engine = 'pymoo'
        elif self.algorithm == 'age':
            self.configure_age()
            self.engine = 'pymoo'
        else: # pragma: no cover
            logger.error(
                '[DyNAS-T] algorithm "{}" not implemented.'.format(self.algorithm)
            )
            raise NotImplementedError

    def configure_nsga2(
        self,
        population: int = 50,
        num_evals: int = 1000,
        warm_pop: np.ndarray = None,
        crossover_prob: float = 0.9,
        crossover_eta: float = 15.0,
        mutation_prob: float = 0.02,
        mutation_eta: float = 20.0,
    ) -> None:
        self.n_gens = num_evals / population

        if type(warm_pop) == 'numpy.ndarray':
            logger.info('[DyNAS-T] Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = get_sampling("int_lhs")

        self.algorithm_def = NSGA2(
            pop_size=population,
            sampling=sample_strategy,
            crossover=get_crossover("int_sbx", prob=crossover_prob, eta=crossover_eta),
            mutation=get_mutation("int_pm", prob=mutation_prob, eta=mutation_eta),
            eliminate_duplicates=True,
        )

    def configure_age(
        self,
        population: int = 50,
        num_evals: int = 1000,
        warm_pop: np.ndarray = None,
        crossover_prob: float = 0.9,
        crossover_eta: float = 15.0,
        mutation_prob: float = 0.02,
        mutation_eta: float = 20.0,
    ) -> None:
        self.engine = 'pymoo'
        self.n_gens = num_evals / population

        if type(warm_pop) == 'numpy.ndarray':
            logger.info('[DyNAS-T] Using warm start population')
            sample_strategy = warm_pop
        else:
            sample_strategy = get_sampling("int_lhs")

        self.algorithm_def = AGEMOEA(
            pop_size=population,
            sampling=sample_strategy,
            crossover=get_crossover("int_sbx", prob=crossover_prob, eta=crossover_eta),
            mutation=get_mutation("int_pm", prob=mutation_prob, eta=mutation_eta),
            eliminate_duplicates=True,
        )

    def run_search(
        self,
        problem: Problem,
        save_history=False,
    ) -> pymoo.core.result.Result:
        """Starts the search process for the algorithm and problem class that have
        been previously defined.
        """

        logger.info('[DyNAS-T] Running Search')
        start_time = time.time()

        # Note: Known issue with the pymoo v0.5.0 save_history=True option
        if self.engine == 'pymoo':
            result = minimize(
                problem,
                self.algorithm_def,
                ('n_gen', int(self.n_gens)),
                seed=self.seed,
                save_history=save_history,
                verbose=self.verbose,
            )
        else: # pragma: no cover
            logger.error('[DyNAS-T] Invalid algorithm engine configuration!')
            raise NotImplementedError

        logger.info(
            '[DyNAS-T] Genetic algorithm search took {:.3f} seconds.'.format(
                time.time() - start_time
            )
        )

        return result


class ProblemMultiObjective(Problem):
    """
    Interface between the user-defined evaluation interface and the SearchAlgoManager.

    Args:
        evaluation_interface : Class
            Class that handles the objective measurement call from the supernet.
        param_count : int
            Number variables in the search space (e.g., OFA MobileNetV3 has 45)
        param_upperbound : array
            The upper int array that defines how many options each design variable has.
    """

    def __init__(
        self,
        evaluation_interface: EvaluationInterface,
        param_count: int,
        param_upperbound: list,
    ):
        super().__init__(
            n_var=param_count,
            n_obj=2,
            n_constr=0,
            xl=0,
            xu=param_upperbound,
            type_var=np.int,
        )

        self.evaluation_interface = evaluation_interface

    def _evaluate(
        self,
        x: list,
        out,
        *args,
        **kwargs,
    ) -> None:

        # Store results for a given generation for PyMoo
        objective_x_arr, objective_y_arr = list(), list()

        # Measure new individuals
        for i in range(len(x)):

            _, objective_x, objective_y = self.evaluation_interface.eval_subnet(x[i])

            objective_x_arr.append(objective_x)
            objective_y_arr.append(objective_y)

        print('.', end='', flush=True)

        # Update PyMoo with evaluation data
        out["F"] = anp.column_stack([objective_x_arr, objective_y_arr])
