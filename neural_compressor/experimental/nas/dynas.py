"""DyNAS approach class."""

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

import os

from deprecated import deprecated

from neural_compressor.conf.config import Conf, NASConfig
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

from .nas import NASBase
from .nas_utils import nas_registry

DyNASManager = LazyImport("dynast.dynast_manager.DyNAS")


@deprecated(version="2.0")
@nas_registry("DyNAS")
class DyNAS(NASBase):
    """DyNAS approach.

    Defining the pipeline for DyNAS approach.

    Args:
        conf_fname_or_obj (string or obj):
            The path to the YAML configuration file or the object of NASConfig.
    """

    def __init__(self, conf_fname_or_obj):
        """Initialize the attributes."""
        super().__init__()

        self.init_cfg(conf_fname_or_obj)

        self.dynas_manager = DyNASManager(
            supernet=self.supernet,
            optimization_metrics=self.metrics,
            measurements=self.metrics,
            search_tactic="linas",
            num_evals=self.num_evals,
            results_path=self.results_csv_path,
            dataset_path=self.dataset_path,
            seed=self.seed,
            population=self.population,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            search_algo=self.search_algo,
            supernet_ckpt_path=self.supernet_ckpt_path,
            dataloader_workers=self.num_workers,
            distributed=self.distributed,
            test_fraction=self.test_fraction,
        )

    def search(self):
        """Execute the search process.

        Returns:
            Best model architectures found in the search process.
        """
        return self.dynas_manager.search()

    def select_model_arch(self):  # pragma: no cover
        """Select the model architecture."""
        # model_arch_proposition intrinsically contained in
        # pymoo.minimize API of search_manager.run_search method,
        # don't have to implement it explicitly.
        pass

    def init_cfg(self, conf_fname_or_obj):
        """Initialize the configuration."""
        logger.info("init_cfg")
        if isinstance(conf_fname_or_obj, str):
            if os.path.isfile(conf_fname_or_obj):
                self.conf = Conf(conf_fname_or_obj).usr_cfg
        elif isinstance(conf_fname_or_obj, NASConfig):
            conf_fname_or_obj.validate()
            self.conf = conf_fname_or_obj.usr_cfg
        else:  # pragma: no cover
            raise NotImplementedError("Please provide a str path to the config file or an object of NASConfig.")
        # self.init_search_cfg(self.conf.nas)
        assert "dynas" in self.conf.nas, "Must specify dynas section."
        dynas_config = self.conf.nas.dynas
        self.seed = self.conf.nas.search.seed
        self.search_algo = self.conf.nas.search.search_algorithm
        self.supernet = dynas_config.supernet
        self.distributed = dynas_config.distributed
        self.metrics = dynas_config.metrics
        self.num_evals = dynas_config.num_evals
        self.results_csv_path = dynas_config.results_csv_path
        self.dataset_path = dynas_config.dataset_path
        self.supernet_ckpt_path = dynas_config.supernet_ckpt_path
        self.batch_size = dynas_config.batch_size
        self.eval_batch_size = dynas_config.eval_batch_size
        self.num_workers = dynas_config.num_workers
        self.test_fraction = dynas_config.test_fraction
        if dynas_config.population < 10:  # pragma: no cover
            raise NotImplementedError("Please specify a population size >= 10")
        else:
            self.population = dynas_config.population
