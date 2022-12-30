"""DyNAS Manager class."""

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
from neural_compressor.utils import logger
from sklearn.model_selection import train_test_split


class ParameterManager:
    """The ParameterManager class.

    It handles the super-network encoding representation and translates between
    the 1-hot predictor, pymoo, and super-network dictionary formats during search.

    Args:
        param_dict (dict): the search space dictionary.
    """

    def __init__(
        self,
        param_dict: dict,
        verbose: bool = False,
        seed: int = 0,
    ) -> None:
        """Initialize the attributes."""
        self.param_dict = param_dict
        self.verbose = verbose
        self.mapper, self.param_upperbound, self.param_count = self.process_param_dict()
        self.inv_mapper = self._inv_mapper()
        self.set_seed(seed)

    def process_param_dict(self) -> Tuple[list, list, int]:
        """Build a parameter mapping arrays and an upper-bound vector for pymoo."""
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
                parameter_mapper.append(
                    dict(zip(index_simple, options['vars'])))

        if self.verbose:  # pragma: no cover
            logger.info(
                '[DyNAS-T] Problem definition variables: {}'.format(
                    parameter_count)
            )
            logger.info(
                '[DyNAS-T] Variable Upper Bound array: {}'.format(
                    parameter_upperbound)
            )
            logger.info(
                '[DyNAS-T] Mapping dictionary created of length: {}'.format(
                    len(parameter_mapper)
                )
            )
            logger.info(
                '[DyNAS-T] Parameter Bound: {}'.format(parameter_bound))

        return parameter_mapper, parameter_upperbound, parameter_count

    def _inv_mapper(self) -> list:
        """Build inverse of self.mapper."""
        inv_parameter_mapper = list()

        for value in self.mapper:
            inverse_dict = {v: k for k, v in value.items()}
            inv_parameter_mapper.append(inverse_dict)

        return inv_parameter_mapper

    def onehot_generic(self, in_array: list) -> np.ndarray:
        """Generate onehot vector.

        This is a generic approach to one-hot vectorization for predictor training
        and testing. It does not account for unused parameter mapping (e.g. block depth).
        For unused parameter mapping, the end user will need to provide a custom solution.

        Args:
            input_array (list): the pymoo individual 1-D vector

        Returns:
            mapper (numpy array): the map for elastic parameters of the supernetwork
        """
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
        """Generate a random subnetwork from the possible elastic parameter range."""
        pymoo_vector = list()
        for i in range(len(self.mapper)):
            options = [x for x in range(len(self.mapper[i]))]
            pymoo_vector.append(random.choice(options))

        return pymoo_vector

    def random_samples(self, size: int = 100, trial_limit: int = 100000) -> List[list]:
        """Generate a list of random subnetworks from the possible elastic parameter range."""
        pymoo_vector_list = list()

        trials = 0
        while len(pymoo_vector_list) < size and trials < trial_limit:
            sample = self.random_sample()
            if sample not in pymoo_vector_list:
                pymoo_vector_list.append(sample)
            trials += 1

        if trials >= trial_limit:
            logger.warning(
                '[DyNAS-T] Unable to create unique list of samples.')

        return pymoo_vector_list

    def translate2param(self, pymoo_vector: list) -> dict:
        """Translate a PyMoo 1-D parameter vector back to the elastic parameter dictionary format."""
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
        """Translate a single parameter dict to pymoo vector."""
        output = list()

        mapper_counter = 0
        for key, value in self.param_dict.items():
            param_counter = 0
            for i in range(value['count']):
                output.append(
                    self.inv_mapper[mapper_counter][parameters[key]
                                                    [param_counter]]
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
        """Import a csv file.

        Import a csv file generated from a supernetwork search for the purpose
        of training a predictor.

        Args:
            filepath (str): path of the csv to be imported.
            config (str): the subnetwork configuration.
            objective (str): target/label for the subnet configuration (e.g. accuracy, latency).
            column_names: a list of column names for the dataframe.

        Returns:
            df: the output dataframe that contains the original config dict, pymoo, and 1-hot
                equivalent vector for training.
        """
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
        """Set the random seed for randomized subnet generation and test/train split."""
        self.seed = seed
        random.seed(seed)

    @staticmethod
    def create_training_set(
        dataframe: pd.DataFrame,
        train_with_all: bool = True,
        split: float = 0.33,
        seed: bool = None,
    ) -> Tuple[list, list, list, list]:
        """Create the training set.

        Create a sklearn compatible test/train set from an imported results csv
        after "import_csv" method is run.
        """
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


class TransformerLTEncoding(ParameterManager):  #noqa: D101
    def __init__(self, param_dict: dict, verbose: bool = False, seed: int = 0):  #noqa: D107
        super().__init__(param_dict, verbose, seed)

    def onehot_custom(self, subnet_cfg, provide_onehot=True):  #noqa: D102

        features = []
        features.extend(subnet_cfg['encoder_embed_dim'])

        encode_layer_num_int = 6

        # Encoder FFN Embed Dim
        encoder_ffn_embed_dim = subnet_cfg['encoder_ffn_embed_dim']

        if encode_layer_num_int < 6:
            encoder_ffn_embed_dim.extend([0]*(6-encode_layer_num_int))
        features.extend(encoder_ffn_embed_dim)

        # Encoder Self-Attn Heads

        encoder_self_attention_heads = subnet_cfg['encoder_self_attention_heads'][:encode_layer_num_int]

        if encode_layer_num_int < 6:
            encoder_self_attention_heads.extend([0]*(6-encode_layer_num_int))
        features.extend(encoder_self_attention_heads)

        features.extend(subnet_cfg['decoder_embed_dim'])

        decoder_layer_num = subnet_cfg['decoder_layer_num']
        decoder_layer_num_int = decoder_layer_num[0]
        features.extend(decoder_layer_num)

        # Decoder FFN Embed Dim
        decoder_ffn_embed_dim = subnet_cfg['decoder_ffn_embed_dim'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
            decoder_ffn_embed_dim.extend([0]*(6-decoder_layer_num_int))
        features.extend(decoder_ffn_embed_dim)

        # Decoder Attn Heads
        decoder_self_attention_heads = subnet_cfg['decoder_self_attention_heads'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
            decoder_self_attention_heads.extend([0]*(6-decoder_layer_num_int))
        features.extend(decoder_self_attention_heads)

        # Decoder ENDE HEADS

        decoder_ende_attention_heads = subnet_cfg['decoder_ende_attention_heads'][:decoder_layer_num_int]

        if decoder_layer_num_int < 6:
            decoder_ende_attention_heads.extend([0]*(6-decoder_layer_num_int))

        features.extend(decoder_ende_attention_heads)

        arbitrary_ende_attn_trans = []
        for i in range(decoder_layer_num_int):
            if subnet_cfg['decoder_arbitrary_ende_attn'][i] == -1:
                arbitrary_ende_attn_trans.append(1)
            elif subnet_cfg['decoder_arbitrary_ende_attn'][i] == 1:
                arbitrary_ende_attn_trans.append(2)
            elif subnet_cfg['decoder_arbitrary_ende_attn'][i] == 2:
                arbitrary_ende_attn_trans.append(3)

        if decoder_layer_num_int < 6:
            arbitrary_ende_attn_trans.extend([0]*(6-decoder_layer_num_int))
        features.extend(arbitrary_ende_attn_trans)

        if provide_onehot == True:
            examples = np.array([features])
            one_hot_count = 0
            unique_values = self.unique_values

            for unique in unique_values:
                one_hot_count += len(unique.tolist())

            one_hot_examples = np.zeros((examples.shape[0], one_hot_count))
            for e, example in enumerate(examples):
                offset = 0
                for f in range(len(example)):
                    index = np.where(unique_values[f] == example[f])[
                        0] + offset
                    one_hot_examples[e, index] = 1.0
                    offset += len(unique_values[f])
            return one_hot_examples

        else:
            return features

    def import_csv(
        self,
        filepath: str,
        config: str,
        objective: str,
        column_names: List[str] = None,
        drop_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Import a csv file generated from a supernetwork search for the purpose of training a predictor.

        filepath - path of the csv to be imported.
        config - the subnetwork configuration
        objective - target/label for the subnet configuration (e.g. accuracy, latency)
        column_names - a list of column names for the dataframe
        df - the output dataframe that contains the original config dict, pymoo, and 1-hot
             equivalent vector for training.
        """
        if column_names == None:
            df = pd.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)
            df.columns = column_names
        df = df[[config, objective]]
        # Old corner case coverage
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
            # Onehot predictor format
            config_as_onehot = self.onehot_custom(
                config_as_dict, provide_onehot=False)
            convert_to_onehot.append(config_as_onehot)
        df[config] = convert_to_dict
        df['config_pymoo'] = convert_to_pymoo
        df['config_onehot'] = convert_to_onehot

        return df

    # @staticmethod
    def create_training_set(
        self,
        dataframe: pd.DataFrame,
        train_with_all: bool = True,
        split: float = 0.33,
        seed: bool = None,
    ) -> Tuple[list, list, list, list]:
        """Create a sklearn compatible test/train.

        The set is created from an imported results csv after "import_csv" method is run.
        """
        collect_rows = list()
        for i in range(len(dataframe)):
            collect_rows.append(np.asarray(dataframe['config_onehot'].iloc[i]))
        features = np.asarray(collect_rows)
        labels = dataframe.drop(
            columns=['config', 'config_pymoo', 'config_onehot']).values

        assert len(features) == len(labels)
        one_hot_count = 0
        unique_values = []

        for c in range(features.shape[1]):
            unique_values.append(np.unique(features[:, c]))
            one_hot_count += len(unique_values[-1])
        one_hot_examples = np.zeros((features.shape[0], one_hot_count))
        for e, example in enumerate(features):
            offset = 0
            for f in range(len(example)):
                index = np.where(unique_values[f] == example[f])[0] + offset
                one_hot_examples[e, index] = 1.0
                offset += len(unique_values[f])

        features = one_hot_examples
        self.unique_values = unique_values
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
