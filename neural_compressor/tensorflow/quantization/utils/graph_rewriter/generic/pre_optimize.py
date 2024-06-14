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
"""Pre Optimization Entrance."""

import copy
import logging

import tensorflow as tf

from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.utils import (
    dump_elapsed_time,
    version1_eq_version2,
    version1_gte_version2,
    version1_lt_version2,
)

from .convert_add_to_biasadd import ConvertAddToBiasAddOptimizer
from .convert_layout import ConvertLayoutOptimizer
from .convert_leakyrelu import ConvertLeakyReluOptimizer
from .convert_nan_to_random import ConvertNanToRandom
from .convert_placeholder_to_const import ConvertPlaceholderToConst
from .dilated_contraction import DilatedContraction
from .dummy_biasadd import InjectDummyBiasAddOptimizer
from .expanddims_optimizer import ExpandDimsOptimizer
from .fetch_weight_from_reshape import FetchWeightFromReshapeOptimizer
from .fold_batch_norm import FoldBatchNormNodesOptimizer
from .fold_constant import GraphFoldConstantOptimizer
from .fuse_biasadd_add import FuseBiasAddAndAddOptimizer
from .fuse_column_wise_mul import FuseColumnWiseMulOptimizer
from .fuse_conv_with_math import FuseConvWithMathOptimizer
from .fuse_decomposed_bn import FuseDecomposedBNOptimizer
from .fuse_decomposed_in import FuseDecomposedINOptimizer
from .fuse_gelu import FuseGeluOptimizer
from .fuse_layer_norm import FuseLayerNormOptimizer
from .fuse_reshape_transpose import FuseTransposeReshapeOptimizer
from .graph_cse_optimizer import GraphCseOptimizer
from .grappler_pass import GrapplerOptimizer
from .move_squeeze_after_relu import MoveSqueezeAfterReluOptimizer
from .remove_training_nodes import RemoveTrainingNodesOptimizer
from .rename_batch_norm import RenameBatchNormOptimizer
from .split_shared_input import SplitSharedInputOptimizer
from .strip_equivalent_nodes import StripEquivalentNodesOptimizer
from .strip_unused_nodes import StripUnusedNodesOptimizer
from .switch_optimizer import SwitchOptimizer


class PreOptimization:
    """Pre optimization for the FP32 models."""

    def __init__(self, model, new_api, device):
        """Initialization."""
        self.model = model
        if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(tf.version.VERSION, "1.15.0-up3"):
            self.optimization = {
                "pruning": True,
                "shape": True,
                "constfold": False,
                "arithmetic": False,
                "dependency": True,
                "debug_stripper": True,
                "loop": True,
            }
        else:  # pragma: no cover
            self.optimization = {
                "pruning": True,
                "shape": True,
                "dependency": True,
                "debug_stripper": True,
                "loop": True,
            }
        # Table initialization should disable grappler dependency and pruning pass
        node_names = [node.name for node in model.graph_def.node]
        if "init_all_tables" in node_names:  # pragma: no cover
            self.optimization["dependency"] = False
            self.optimization["pruning"] = False
        self.new_api = new_api
        self.device = device
        self.analyzer = GraphAnalyzer()
        self.analyzer.graph = model.graph_def
        self.analyzer.parse_graph()
        self._tmp_graph_def = None
        self._excluded_node_names = []

    def get_excluded_node_names(self):
        """Get the excluded node name.

        Returns:
            string list: the excluded ops' name
        """
        return self._excluded_node_names

    @dump_elapsed_time("Pass Pre Optimization")
    def get_optimized_model(self, itex_mode=False):
        """Executed the non-precision dependent graph optimization.

        The input graph will be optimized with following passes:
        1. Remove the training nodes like Identity Op.
        2. Split the shared nodes like weights node for multi-Conv2d.
        3. Fold Constant Nodes as less as possible.
        4. Fuse the Mul node into the previous Conv2D/MatMul if possible.
        5. Strip the useless nodes.
        6. Do the Common sequence elimation optimization on the graph.
        7. Fold the BN node into the previous Conv2D if possible.

        Returns:
            [graphdef]: the optimized graphdef object.
        """
        from neural_compressor.tensorflow.utils import Model

        origin_model = Model(self.model._model, **self.model.kwargs, backend="itex" if itex_mode else "default")
        origin_model.name = self.model.name
        origin_model.model_type = self.model.model_type
        origin_model.output_tensor_names = self.model.output_tensor_names
        origin_model.input_tensor_names = self.model.input_tensor_names
        origin_model.workspace_path = self.model.workspace_path

        output_node_names = self.model.output_node_names
        input_node_names = self.model.input_node_names
        input_output_names = output_node_names + input_node_names

        # Add device info before convert layout
        # Google in layout optimizer where all nodes in the graph are expected to have their device
        # information set (earlier version < 2.10.0 this was not needed).
        if version1_gte_version2(tf.version.VERSION, "2.10.0"):
            cur_graph = GraphAnalyzer()
            cur_graph.graph = self.model.graph_def
            graph_info = cur_graph.parse_graph()

            if self.device == "cpu":
                cpus = tf.config.list_physical_devices("CPU")
                node_device = cpus[0].name.replace("physical_device:", "")
            else:  # pragma: no cover
                gpus = tf.config.list_physical_devices("GPU")
                if len(gpus) == 0:
                    xpus = tf.config.list_physical_devices("XPU")
                    if len(xpus) == 0:
                        cpus = tf.config.list_physical_devices("CPU")
                        node_device = cpus[0].name.replace("physical_device:", "")
                    else:
                        node_device = xpus[0].name.replace("physical_device:", "")
                else:
                    node_device = gpus[0].name.replace("physical_device:", "")
            for node_name in list(graph_info.keys()):
                node = graph_info[node_name].node
                node.device = node_device
            self._tmp_graph_def = cur_graph.dump_graph()

            self._tmp_graph_def = ConvertLayoutOptimizer(self._tmp_graph_def, output_node_names).do_transformation()
        else:
            self._tmp_graph_def = ConvertLayoutOptimizer(self.model.graph_def, output_node_names).do_transformation()

        self._tmp_graph_def = ConvertPlaceholderToConst(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = SwitchOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = GrapplerOptimizer(
            self._tmp_graph_def, input_output_names, self.optimization
        ).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def, input_node_names, output_node_names
        ).do_transformation()

        self._tmp_graph_def = RemoveTrainingNodesOptimizer(
            self._tmp_graph_def, protected_nodes=input_output_names
        ).do_transformation()

        self._tmp_graph_def = SplitSharedInputOptimizer(self._tmp_graph_def).do_transformation()

        # Put FuseDecomposedBNOptimizer before GraphFoldConstantOptimizer
        # The 'Sub' op in the small decomposed ops of BN will be converted to const by GraphFoldConstantOptimizer.
        # Then the FuseDecomposedBNOptimizer can't fuse the small decomposed ops to BN.
        if self.new_api:
            self._tmp_graph_def = FuseDecomposedBNOptimizer(self._tmp_graph_def).do_transformation()
            self._tmp_graph_def = FuseDecomposedINOptimizer(self._tmp_graph_def).do_transformation()
            self._tmp_graph_def = FuseLayerNormOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = GraphFoldConstantOptimizer(self._tmp_graph_def).do_transformation()

        if not self.new_api:
            self._tmp_graph_def = FuseDecomposedBNOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseColumnWiseMulOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripUnusedNodesOptimizer(
            self._tmp_graph_def, input_node_names, output_node_names
        ).do_transformation()

        self._tmp_graph_def = FuseGeluOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = GraphCseOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = RenameBatchNormOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = ConvertLeakyReluOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = ConvertAddToBiasAddOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseTransposeReshapeOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FuseConvWithMathOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = ExpandDimsOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = FetchWeightFromReshapeOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = MoveSqueezeAfterReluOptimizer(self._tmp_graph_def).do_transformation()

        if not self.new_api and not itex_mode:
            # TODO we need to remove below optimizer once the TF enabled the single
            # matmul op quantization
            self._tmp_graph_def = InjectDummyBiasAddOptimizer(
                self._tmp_graph_def, output_node_names
            ).do_transformation()

        self._tmp_graph_def = FuseBiasAddAndAddOptimizer(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = ConvertNanToRandom(self._tmp_graph_def).do_transformation()

        self._tmp_graph_def = StripEquivalentNodesOptimizer(self._tmp_graph_def, output_node_names).do_transformation()

        if self.new_api or itex_mode:
            self._tmp_graph_def = DilatedContraction(self._tmp_graph_def).do_transformation()

        # node device info will be removed by GrapplerOptimizer, insert it again.
        if version1_lt_version2(tf.version.VERSION, "2.0.0"):  # pragma: no cover
            from tensorflow._api.v1.config import experimental

            list_physical_devices = experimental.list_physical_devices
        else:
            list_physical_devices = tf.config.list_physical_devices
        cur_graph = GraphAnalyzer()
        cur_graph.graph = self._tmp_graph_def
        graph_info = cur_graph.parse_graph()

        if self.device == "cpu":
            cpus = list_physical_devices("CPU")
            node_device = cpus[0].name.replace("physical_device:", "")
        else:  # pragma: no cover
            gpus = list_physical_devices("GPU")
            if len(gpus) == 0:
                xpus = list_physical_devices("XPU")
                if len(xpus) == 0:
                    cpus = list_physical_devices("CPU")
                    node_device = cpus[0].name.replace("physical_device:", "")
                else:
                    node_device = xpus[0].name.replace("physical_device:", "")
            else:
                node_device = gpus[0].name.replace("physical_device:", "")
        for node_name in list(graph_info.keys()):
            node = graph_info[node_name].node
            node.device = node_device
        self._tmp_graph_def = cur_graph.dump_graph()

        self._tmp_graph_def.library.CopyFrom(self.model.graph_def.library)

        for function_def in self.model.graph_def.library.function:
            if function_def.signature.name == "swish_f32":  # pragma: no cover
                self._tmp_graph_def.library.function.extend([copy.deepcopy(function_def)])

        origin_model.graph_def = self._tmp_graph_def
        return origin_model

    def get_matched_nodes(self, patterns):
        """Search the matched nodes with the specified patterns.

        Args:
            patterns ([string list]): The patterns should be illustrated as below.
                [['MatMul'], ("BiasAdd"), ("Relu",)]

        Returns:
            [string list]: It will return the list that contains the matched nodes name
                and pattern. ['matched_node_a_name', 'matched_node_a_name',['MatMul','BiasAdd']]
        """
        self.analyzer.graph = self._tmp_graph_def
        self.analyzer.parse_graph()
        res = []

        for sub_pattern in patterns:
            res.extend([i for i in self.analyzer.query_fusion_pattern_nodes(sub_pattern) if i not in res])
        return res

    def has_positive_input(self, node_name):
        """Check the specified node has the positive input or not.

        Args:
            node_name ([string]): node's name

        Returns:
            [bool]: True if the node has the positive input data,
                    False if the node has the negative input data.
        """
        return self.analyzer.has_positive_input(node_name)
