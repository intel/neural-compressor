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

import ast
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural_compressor.utils import logger


class ParameterManager:
    """The ParameterManager class handles the super-network encoding representation and translates between
    the 1-hot predictor, pymoo, and super-network dictionary formats during search.
    """

    def __init__(
        self,
        param_dict: dict,
        verbose: bool = False,
        seed: int = 0,
    ) -> None:
        self.param_dict = param_dict
        self.verbose = verbose
        self.mapper, self.param_upperbound, self.param_count = self.process_param_dict()
        self.inv_mapper = self._inv_mapper()
        self.set_seed(seed)

    def process_param_dict(self) -> Tuple[list, list, int]:
        '''
        Builds a parameter mapping arrays and an upper-bound vector for pymoo.
        '''
        parameter_count = 0
        parameter_bound = list()
        parameter_upperbound = list()
        parameter_mapper = list()

        for parameter, options in self.param_dict.items():
            # How many variables should be searched for
            parameter_count += options['count']
            parameter_bound.append(options['count'])

            # How many variables for each parameter
            for i in range(options['count']):
                parameter_upperbound.append(len(options['vars']) - 1)
                index_simple = [x for x in range(len(options['vars']))]
                parameter_mapper.append(dict(zip(index_simple, options['vars'])))

        if self.verbose: # pragma: no cover
            logger.info(
                '[DyNAS-T] Problem definition variables: {}'.format(parameter_count)
            )
            logger.info(
                '[DyNAS-T] Variable Upper Bound array: {}'.format(parameter_upperbound)
            )
            logger.info(
                '[DyNAS-T] Mapping dictionary created of length: {}'.format(
                    len(parameter_mapper)
                )
            )
            logger.info('[DyNAS-T] Parameter Bound: {}'.format(parameter_bound))

        return parameter_mapper, parameter_upperbound, parameter_count

    def _inv_mapper(self) -> list:
        '''
        Builds inverse of self.mapper
        '''
        inv_parameter_mapper = list()

        for value in self.mapper:
            inverse_dict = {v: k for k, v in value.items()}
            inv_parameter_mapper.append(inverse_dict)

        return inv_parameter_mapper

    def onehot_generic(self, in_array: list) -> np.ndarray:
        '''
        This is a generic approach to one-hot vectorization for predictor training
        and testing. It does not account for unused parameter mapping (e.g. block depth).
        For unused parameter mapping, the end user will need to provide a custom solution.

        input_array - the pymoo individual 1-D vector
        mapper - the map for elastic parameters of the supernetwork
        '''
        # Insure compatible array and mapper
        assert len(in_array) == len(self.mapper)

        onehot = list()

        # This function converts a pymoo input vector to a one-hot feature vector
        for i in range(len(self.mapper)):
            segment = [0 for _ in range(len(self.mapper[i]))]
            segment[in_array[i]] = 1
            onehot.extend(segment)

        return np.array(onehot)

    def random_sample(self) -> list:
        '''
        Generates a random subnetwork from the possible elastic parameter range
        '''
        pymoo_vector = list()
        for i in range(len(self.mapper)):
            options = [x for x in range(len(self.mapper[i]))]
            pymoo_vector.append(random.choice(options))

        return pymoo_vector

    def random_samples(self, size: int = 100, trial_limit: int = 100000) -> List[list]:
        '''
        Generates a list of random subnetworks from the possible elastic parameter range
        '''
        pymoo_vector_list = list()

        trials = 0
        while len(pymoo_vector_list) < size and trials < trial_limit:
            sample = self.random_sample()
            if sample not in pymoo_vector_list:
                pymoo_vector_list.append(sample)
            trials += 1

        if trials >= trial_limit:
            logger.warning('[DyNAS-T] Unable to create unique list of samples.')

        return pymoo_vector_list

    def translate2param(self, pymoo_vector: list) -> dict:
        '''
        Translate a PyMoo 1-D parameter vector back to the elastic parameter dictionary format
        '''
        output = dict()

        # Assign (and map) each vector element to the appropriate parameter dictionary key
        counter = 0
        for key, value in self.param_dict.items():
            output[key] = list()
            for i in range(value['count']):
                output[key].append(self.mapper[counter][pymoo_vector[counter]])
                counter += 1

        # Insure correct vector mapping occurred
        assert counter == len(self.mapper)

        return output

    def translate2pymoo(self, parameters: dict) -> list:
        '''
        Translate a single parameter dict to pymoo vector
        '''
        output = list()

        mapper_counter = 0
        for key, value in self.param_dict.items():
            param_counter = 0
            for i in range(value['count']):
                output.append(
                    self.inv_mapper[mapper_counter][parameters[key][param_counter]]
                )
                mapper_counter += 1
                param_counter += 1

        return output

    def import_csv(
        self,
        filepath: str,
        config: str,
        objective: str,
        column_names: List[str] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        '''
        Import a csv file generated from a supernetwork search for the purpose
        of training a predictor.

        filepath - path of the csv to be imported.
        config - the subnetwork configuration
        objective - target/label for the subnet configuration (e.g. accuracy, latency)
        column_names - a list of column names for the dataframe
        df - the output dataframe that contains the original config dict, pymoo, and 1-hot
             equivalent vector for training.
        '''

        if column_names == None:
            df = pd.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)
            df.columns = column_names
        df = df[[config, objective]]

        # OFA corner case coverage
        df[config] = df[config].replace({'null': 'None'}, regex=True)

        if drop_duplicates:
            df.drop_duplicates(subset=[config], inplace=True)
            df.reset_index(drop=True, inplace=True)

        convert_to_dict = list()
        convert_to_pymoo = list()
        convert_to_onehot = list()
        for i in range(len(df)):
            # Elastic Param Config format
            config_as_dict = ast.literal_eval(df[config].iloc[i])
            convert_to_dict.append(config_as_dict)
            # PyMoo 1-D vector format
            config_as_pymoo = self.translate2pymoo(config_as_dict)
            convert_to_pymoo.append(config_as_pymoo)
            # Onehot preditor format
            config_as_onehot = self.onehot_generic(config_as_pymoo)
            convert_to_onehot.append(config_as_onehot)

        df[config] = convert_to_dict
        df['config_pymoo'] = convert_to_pymoo
        df['config_onehot'] = convert_to_onehot

        return df

    def set_seed(self, seed) -> None:
        '''
        Set the random seed for randomized subnet generation and test/train split
        '''
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def create_training_set(
        dataframe: pd.DataFrame,
        train_with_all: bool = True,
        split: float = 0.33,
        seed: bool = None,
    ) -> Tuple[list, list, list, list]:
        '''
        Create a sklearn compatible test/train set from an imported results csv
        after "import_csv" method is run.
        '''

        collect_rows = list()
        for i in range(len(dataframe)):
            collect_rows.append(np.asarray(dataframe['config_onehot'].iloc[i]))
        features = np.asarray(collect_rows)

        labels = dataframe.drop(
            columns=['config', 'config_pymoo', 'config_onehot']
        ).values

        assert len(features) == len(labels)

        if train_with_all:
            logger.info('[DyNAS-T] Training set size={}'.format(len(labels)))
            return features, labels
        else:
            features_train, features_test, labels_train, labels_test = train_test_split(
                features, labels, test_size=split, random_state=seed
            )
            logger.info(
                '[DyNAS-T] Test ({}) Train ({}) ratio is {}.'.format(
                    len(labels_train), len(labels_test), split
                )
            )
            return features_train, features_test, labels_train, labels_test
