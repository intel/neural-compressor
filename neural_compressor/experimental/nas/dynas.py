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

import pandas as pd
from neural_compressor.conf.config import Conf, NASConfig
from neural_compressor.utils import logger

from .nas import NASBase
from .nas_utils import nas_registry


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
        from .dynast.dynas_manager import (ParameterManager,
                                           TransformerLTEncoding)
        from .dynast.dynas_predictor import Predictor
        from .dynast.dynas_search import (ProblemMultiObjective,
                                          SearchAlgoManager)
        from .dynast.dynas_utils import (EvaluationInterfaceMobileNetV3,
                                         EvaluationInterfaceResNet50,
                                         EvaluationInterfaceTransformerLT,
                                         OFARunner, TransformerLTRunner)

        self.ParameterManager = ParameterManager
        self.Predictor = Predictor
        self.ProblemMultiObjective = ProblemMultiObjective
        self.SearchAlgoManager = SearchAlgoManager
        self.SUPERNET_PARAMETERS = {
            'ofa_resnet50': {
                'd':  {'count': 5,  'vars': [0, 1, 2]},
                'e':  {'count': 18, 'vars': [0.2, 0.25, 0.35]},
                'w':  {'count': 6,  'vars': [0, 1, 2]},
            },
            'ofa_mbv3_d234_e346_k357_w1.0': {
                'ks':  {'count': 20, 'vars': [3, 5, 7]},
                'e':  {'count': 20, 'vars': [3, 4, 6]},
                'd':  {'count': 5,  'vars': [2, 3, 4]},
            },
            'ofa_mbv3_d234_e346_k357_w1.2': {
                'ks':  {'count': 20, 'vars': [3, 5, 7]},
                'e':  {'count': 20, 'vars': [3, 4, 6]},
                'd':  {'count': 5,  'vars': [2, 3, 4]},
            },
            'transformer_lt_wmt_en_de': {
                'encoder_embed_dim': {'count': 1, 'vars': [640, 512]},
                'decoder_embed_dim': {'count': 1, 'vars': [640, 512]},
                'encoder_ffn_embed_dim': {'count': 6, 'vars': [3072, 2048, 1024]},
                'decoder_ffn_embed_dim': {'count': 6, 'vars': [3072, 2048, 1024]},
                'decoder_layer_num': {'count': 1, 'vars': [6, 5, 4, 3, 2, 1]},
                'encoder_self_attention_heads': {'count': 6, 'vars': [8, 4]},
                'decoder_self_attention_heads': {'count': 6, 'vars': [8, 4]},
                'decoder_ende_attention_heads': {'count': 6, 'vars': [8, 4]},
                'decoder_arbitrary_ende_attn': {'count': 6, 'vars': [-1, 1, 2]},
            },
        }
        self.RUNNERS = {
            'ofa_resnet50': OFARunner,
            'ofa_mbv3_d234_e346_k357_w1.0': OFARunner,
            'ofa_mbv3_d234_e346_k357_w1.2': OFARunner,
            'transformer_lt_wmt_en_de': TransformerLTRunner,
        }

        self.EVALUATION_INTERFACE = {'ofa_resnet50': EvaluationInterfaceResNet50,
                                     'ofa_mbv3_d234_e346_k357_w1.0': EvaluationInterfaceMobileNetV3,
                                     'ofa_mbv3_d234_e346_k357_w1.2': EvaluationInterfaceMobileNetV3,
                                     'transformer_lt_wmt_en_de': EvaluationInterfaceTransformerLT}

        self.LINAS_INNERLOOP_EVALS = {'ofa_resnet50': 5000,
                                      'ofa_mbv3_d234_e346_k357_w1.0': 20000,
                                      'ofa_mbv3_d234_e346_k357_w1.2': 20000,
                                      'transformer_lt_wmt_en_de': 10000}

        self.SUPERNET_ENCODING = {
            'ofa_resnet50': ParameterManager,
            'ofa_mbv3_d234_e346_k357_w1.0': ParameterManager,
            'ofa_mbv3_d234_e346_k357_w1.2': ParameterManager,
            'ofa_proxyless_d234_e346_k357_w1.3': ParameterManager,
            'transformer_lt_wmt_en_de': TransformerLTEncoding,
        }

        super().__init__()
        self.acc_predictor = None
        self.macs_predictor = None
        self.latency_predictor = None
        self.results_csv_path = None
        self.num_workers = None
        self.init_cfg(conf_fname_or_obj)

    def estimate(self, individual):
        """Estimate performance of the model.

        Returns:
            Evaluated metrics of the model.
        """
        self.validation_interface.eval_subnet(individual)

    def init_for_search(self):
        """Initialize the search configuration."""
        self.supernet_manager = self.SUPERNET_ENCODING[self.supernet](
            param_dict=self.SUPERNET_PARAMETERS[self.supernet], seed=self.seed
        )

        # Validation High-Fidelity Measurement Runner
        self.runner_validate = self.RUNNERS[self.supernet](
            supernet=self.supernet,
            acc_predictor=None,
            macs_predictor=None,
            latency_predictor=None,
            datasetpath=self.dataset_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            checkpoint_path=self.supernet_ckpt_path,
        )

        # Setup validation interface
        self.validation_interface = self.EVALUATION_INTERFACE[self.supernet](
            evaluator=self.runner_validate,
            metrics=self.metrics,
            manager=self.supernet_manager,
            csv_path=self.results_csv_path
        )

        # Clear csv file if one exists
        # self.validation_interface.clear_csv()

    def search(self):
        """Execute the search process.

        Returns:
            Best model architectures found in the search process.
        """
        self.init_for_search()

        # Randomly sample search space for initial population
        # if number of results in results_csv_path smaller than population.

        if not os.path.exists(self.results_csv_path):
            # Clear also creates empty CSV file.
            self.validation_interface.clear_csv()

        df = pd.read_csv(self.results_csv_path)
        latest_population = [self.supernet_manager.random_sample()
                             for _ in range(max(self.population - df.shape[0], 0))]

        # Start Lightweight Iterative Neural Architecture Search (LINAS)
        num_loops = round(self.num_evals/self.population)
        for loop in range(num_loops):

            for i, individual in enumerate(latest_population):
                logger.info(
                '[DyNAS-T] Starting eval {} of {} in LINAS loop {} of {}.'.format(
                    i+1, len(latest_population), loop+1, num_loops))
                self.validation_interface.eval_subnet(individual)

            self.create_acc_predictor()
            self.create_macs_predictor()
            self.create_latency_predictor()

            # Inner-loop Low-Fidelity Predictor Runner, need to re-instantiate every loop
            runner_predict = self.RUNNERS[self.supernet](
                supernet=self.supernet,
                acc_predictor=self.acc_predictor,
                macs_predictor=self.macs_predictor,
                latency_predictor=self.latency_predictor,
                datasetpath=self.dataset_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                checkpoint_path=self.supernet_ckpt_path
            )

            # Setup validation interface
            prediction_interface = self.EVALUATION_INTERFACE[self.supernet](
                evaluator=runner_predict,
                manager=self.supernet_manager,
                metrics=self.metrics,
                csv_path=None,
                predictor_mode=True
            )

            problem = self.ProblemMultiObjective(
                evaluation_interface=prediction_interface,
                param_count=self.supernet_manager.param_count,
                param_upperbound=self.supernet_manager.param_upperbound
            )

            if self.search_algo == 'age':
                search_manager = self.SearchAlgoManager(
                    algorithm='age', seed=self.seed)
                search_manager.configure_age(population=self.population,
                                             num_evals=self.LINAS_INNERLOOP_EVALS[self.supernet])
            else:
                search_manager = self.SearchAlgoManager(
                    algorithm='nsga2', seed=self.seed)
                search_manager.configure_nsga2(population=self.population,
                                               num_evals=self.LINAS_INNERLOOP_EVALS[self.supernet])

            results = search_manager.run_search(problem)

            latest_population = results.pop.get('X')

        logger.info(
            "[DyNAS-T] Validated model architectures in file: {}".format(self.results_csv_path))

        output = list()
        for individual in latest_population:
            output.append(self.supernet_manager.translate2param(individual))

        return output

    def select_model_arch(self): # pragma: no cover
        """Select the model architecture."""
        # model_arch_proposition intrinsically contained in
        # pymoo.minimize API of search_manager.run_search method,
        # don't have to implement it explicitly.
        pass

    def create_acc_predictor(self):
        """Create the accuracy predictor."""
        if 'acc' in self.metrics:
            logger.info('[DyNAS-T] Building Accuracy Predictor')
            df = self.supernet_manager.import_csv(self.results_csv_path,
                                                  config='config',
                                                  objective='acc',
                                                  column_names=['config', 'date', 'lat', 'macs', 'acc'])
            features, labels = self.supernet_manager.create_training_set(df)
            self.acc_predictor = self.Predictor()
            self.acc_predictor.train(features, labels.ravel())
        else:
            self.acc_predictor = None

    def create_macs_predictor(self):
        """Create the MACs predictor."""
        if 'macs' in self.metrics:
            logger.info('[DyNAS-T] Building MACs Predictor')
            df = self.supernet_manager.import_csv(self.results_csv_path,
                                                  config='config',
                                                  objective='macs',
                                                  column_names=['config', 'date', 'lat', 'macs', 'acc'])
            features, labels = self.supernet_manager.create_training_set(df)
            self.macs_predictor = self.Predictor()
            self.macs_predictor.train(features, labels.ravel())
        else:
            self.macs_predictor = None

    def create_latency_predictor(self):
        """Create the latency predictor."""
        if 'lat' in self.metrics:
            logger.info('[DyNAS-T] Building Latency Predictor')
            df = self.supernet_manager.import_csv(self.results_csv_path,
                                                  config='config',
                                                  objective='lat',
                                                  column_names=['config', 'date', 'lat', 'macs', 'acc'])
            features, labels = self.supernet_manager.create_training_set(df)
            self.latency_predictor = self.Predictor()
            self.latency_predictor.train(features, labels.ravel())
        else:
            self.latency_predictor = None

    def init_cfg(self, conf_fname_or_obj):
        """Initialize the configuration."""
        if isinstance(conf_fname_or_obj, str):
            if os.path.isfile(conf_fname_or_obj):
                self.conf = Conf(conf_fname_or_obj).usr_cfg
        elif isinstance(conf_fname_or_obj, NASConfig):
            conf_fname_or_obj.validate()
            self.conf = conf_fname_or_obj.usr_cfg
        else:  # pragma: no cover
            raise NotImplementedError(
                "Please provide a str path to the config file or an object of NASConfig."
            )
        # self.init_search_cfg(self.conf.nas)
        assert 'dynas' in self.conf.nas, "Must specify dynas section."
        dynas_config = self.conf.nas.dynas
        self.search_algo = self.conf.nas.search.search_algorithm
        self.supernet = dynas_config.supernet
        self.metrics = dynas_config.metrics
        self.num_evals = dynas_config.num_evals
        self.results_csv_path = dynas_config.results_csv_path
        self.dataset_path = dynas_config.dataset_path
        self.supernet_ckpt_path = dynas_config.supernet_ckpt_path
        self.batch_size = dynas_config.batch_size
        self.num_workers = dynas_config.num_workers
        if dynas_config.population < 10:  # pragma: no cover
            raise NotImplementedError(
                "Please specify a population size >= 10"
            )
        else:
            self.population = dynas_config.population
