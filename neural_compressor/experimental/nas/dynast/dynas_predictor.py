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

import pickle

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn import linear_model, svm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV


class Predictor:

    DEFAULT_ALPHAS = np.arange(0.1, 10.1, 0.1)
    DEFAULT_COST_FACTORS = np.arange(1.0, 101.0, 1.0)
    DEFAULT_MAX_ITERATIONS = 1000000

    def __init__(self, alphas=DEFAULT_ALPHAS, cost_factors=DEFAULT_COST_FACTORS,
                 max_iterations=DEFAULT_MAX_ITERATIONS, verbose=False):

        SEARCHER_VERBOSITY = 10

        # Initialize label normalization factor
        self.normalization_factor = 1.0

        # Initialize best searcher index
        self.best_index = 0

        # Create lists of regressors and associated hyper-parameters
        regressors = [linear_model.Ridge(max_iter=max_iterations),
                      svm.SVR(kernel='rbf', gamma='auto', epsilon=0.0, max_iter=max_iterations)]
        hyper_parameters = [{'alpha': alphas}, {'C': cost_factors}]

        # Create list of hyper-parameter searchers
        self.searchers = []
        for regressor, parameters in zip(regressors, hyper_parameters):
            self.searchers.append(GridSearchCV(estimator=regressor, param_grid=parameters, n_jobs=-1,
            scoring='neg_mean_absolute_percentage_error', verbose=SEARCHER_VERBOSITY if (verbose) else 0))

    def train(self, examples, labels):

        '''
        Trains the predictor on the specified examples and labels using the underlying regressor.
        Parameters
        ----------
            examples: Examples to be used for training.
            labels: Labels to be used for training.
        Returns
        -------
            None
        '''

        # Compute normalized labels
        self.normalization_factor = 10 ** (np.floor(np.log10(np.amax(labels))) - 1.0)
        normalized_labels = labels / self.normalization_factor

        # Train regressors with optimal parameters
        scores = np.zeros(len(self.searchers))
        for s in range(len(self.searchers)):
            self.searchers[s].fit(examples, normalized_labels)
            scores[s] = self.searchers[s].best_score_

        # Determine index of best searcher
        self.best_index = np.argmax(scores)

    def predict(self, examples):
        '''
        Predicts the output values of the specified examples using the underlying regressor.
        Parameters
        ----------
            examples: Examples for which predictions will be made.
        Returns
        -------
            Predictions of the specified examples.
        '''

        # Compute predictions
        regressor = self.searchers[self.best_index].best_estimator_
        normalized_predictions = regressor.predict(examples)
        predictions = normalized_predictions * self.normalization_factor

        return predictions

    def get_parameters(self):
        '''
        Returns the optimal parameter values of the underlying regressor.
        Parameters
        ----------
            None
        Returns
        -------
            Optimal parameter values of the underlying regressor.
        '''

        # Retrieve optimal parameters
        parameters = {}
        for searcher in self.searchers:
            regressor_name = searcher.best_estimator_.__class__.__name__
            for key in searcher.best_params_:
                parameter_key = regressor_name + '_' + key
                parameters[parameter_key.lower()] = searcher.best_params_[key]

        return parameters

    def get_metrics(self, examples, labels):
        '''
        Computes the performance metrics of the underlying regressor.
        Parameters
        ----------
            examples: Examples to use when computing performance metrics.
            labels: Labels to use when computing performance metrics.
        Returns
        -------
            Performance metrics of the underlying regressor. The metrics are
                Mean absolute percentage error (MAPE)
                Root mean squared error (RMSE)
                Kendall rank correlation coefficient (kendall)
                Spearman's rank correlation coefficient (spearman)
        '''

        # Compute predictions of specified examples
        predictions = self.predict(examples)

        # Compute performance metrics
        mape = 100.0 * mean_absolute_percentage_error(labels, predictions)
        rmse = mean_squared_error(labels, predictions, squared=False)
        kendall, _ = kendalltau(labels, predictions)
        spearman, _ = spearmanr(labels, predictions)

        return mape, rmse, kendall, spearman

    def load(self, filename):
        '''
        Loads the model of the underlying regressor and searcher.
        Parameters
        ----------
            filename: Name of the file from which to load the model.
        Returns
        -------
            None
        '''

        # Load searcher and regressor from specified file
        with open(filename, 'rb') as input_file:
            self.normalization_factor = pickle.load(input_file)
            self.best_index = pickle.load(input_file)
            self.searchers = pickle.load(input_file)

    def save(self, filename):
        '''
        Saves the model of the underlying regressor and searcher.
        Parameters
        ----------
            filename: Name of the file to which to save the model.
        Returns
        -------
            None
        '''

        # Save searcher and regressor to specified file
        with open(filename, 'wb') as output_file:
            pickle.dump(self.normalization_factor, output_file)
            pickle.dump(self.best_index, output_file)
            pickle.dump(self.searchers, output_file)
