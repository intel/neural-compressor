#!/usr/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import random
import json
import sys
import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import tensorflow as tf
from sklearn import metrics

def calc_f1(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro")

def construct_placeholders(num_classes):
    # Define placeholders
    tf.compat.v1.disable_eager_execution()
    placeholders = {
        'labels' : tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch_size' : tf.compat.v1.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    # (G, 
    #         id_map,
    #         placeholders, 
    #         class_map,
    #         num_classes,
    #         batch_size=FLAGS.batch_size,
    #         max_degree=FLAGS.max_degree, 
    #         context_pairs = context_pairs)
    def __init__(self, G, id2idx, 
            placeholders, label_map, num_classes, 
            batch_size=100, max_degree=25,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes
        self.test_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['test']]

    def _make_label_vec(self, node):
            label = self.label_map[node]
            if isinstance(label, list):
                label_vec = np.array(label)
            else:
                label_vec = np.zeros((self.num_classes))
                class_ind = self.label_map[node]
                label_vec[class_ind] = 1
            return label_vec
    def batch_feed_dict(self, batch_nodes, val=False):
            batch1id = batch_nodes
            batch1 = [self.id2idx[n] for n in batch1id]
                
            labels = np.vstack([self._make_label_vec(node) for node in batch1id])
            feed_dict = dict()
            feed_dict.update({'batch1:0': batch1})
            feed_dict.update({'batch_size:0' : len(batch1)})
            return feed_dict, labels


    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset
