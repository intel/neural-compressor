"""Analyze."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import re

from ....utils.utility import LazyImport
from ..utils import logger

torch = LazyImport("torch")
tf = LazyImport("tensorflow")

JIT_SUPPORT_OPS = ["linear", "dropout", "gelu", "silu", "relu", "mul", "add"]

# MHA_SUPPORT_NAMES = ["q", "k", "v"]


def get_attributes(module, attrs: str):
    """Get a multi-level descent module of module.

    Args:
        module (torch.nn.Module): The torch module.
        attrs (str): The attributes' calling path.

    Returns:
        attr: The target attribute of the module.
    """
    assert isinstance(module, torch.nn.Module)
    attrs_list = attrs.split(".")
    sub_module = module
    while attrs_list:
        attr = attrs_list.pop(0)
        sub_module = getattr(sub_module, attr)
    return sub_module


def get_common_module(layer1: str, layer2: str):
    """Get the module which contains layer1 and layer2 (nearest father nodes)"""
    attribute_seq1 = layer1.split(".")
    attribute_seq2 = layer2.split(".")
    target_module = []
    for idx in range(min(len(attribute_seq1), len(attribute_seq2))):
        if attribute_seq1[idx] != attribute_seq2[idx]:
            break
        else:
            target_module.append(attribute_seq1[idx])
    return ".".join(target_module)


def print_iterables(data_iters):
    """Print the auto slim logs."""
    for data in data_iters:
        try:
            logger.info(f"{data}: {data_iters[data]}")  # dict
        except:
            logger.info(f"{data}")  # list


class RecipeSearcher(object):
    """Searcher class which searches patterns with a pre-defined recipe.

    A Recipe is a dict type data which contains the root module's name and
    its sub-modules' levelwise calling way.
    For example, for the self-attention module in Huggingface bert-model,
    if we want to obtain its linear ops (query, key, value and output),
    the recipe should be like:
    recipe_samples = {
        'BertAttention': ["self.query", "self.key", "self.value", "output.dense"]
    }

    Args:
        model (torch.nn.Module): The PyTorch model for searching.
        recipe (dict): A dict containing information of the searching pattern.

    Attributes:
        model: The PyTorch model for searching.
        recipe: A dict containing information of the searching pattern.
        targets: The basic module's name which contains searching pattern.
        searching_results: The list/dict which store matched patterns.
    """

    def __init__(self, model, recipe: dict):
        """Initialize the attributes."""
        assert isinstance(model, torch.nn.Module)
        if "PyTorchFXModel" in type(model).__name__:
            # neural compressor built-in model type
            self.model = model.model
        else:
            self.model = model
        self.recipe = recipe
        self.targets = list(self.recipe.keys())
        self.search_results = []

    def search(self, target_name):
        """Operations called for entire searching process."""
        self.search_results.clear()
        self.dfs_search(self.model, type(self.model).__name__, target_name)
        return self.search_results

    def dfs_search(self, module, module_name, target_name):
        """Operations called for one single search step."""
        module_type = type(module).__name__
        if module_type in self.targets:
            sublayers = [get_attributes(module, sublayer_name) for sublayer_name in self.recipe[module_type]]
            self.search_results.append(sublayers)
        # recursively search
        for n, m in module.named_children():
            self.dfs_search(m, n, target_name)


class JitBasicSearcher(object):
    """Static graph searcher class which searches patterns with PyTorch static graph and its input/output information.

    By converting a PyTorch Model into a static version using torch.jit.trace()/script(),
    we can trace some special pattern in the model and optimize them automatically.
    This class provide some basic functions for jit searcher
    Including generating dummy inputs, generating static graph, analyzing static graph.

    Args:
        model (torch.nn.Module): The PyTorch model for searching.

    Attributes:
        model: The PyTorch model for searching.
        device: The model's current device type.
        static_graph: The static graph of original model.
        flatten_static_graph: A list of string with the model's static graph inference details.
        target_layers: The layer types the searcher will extract.
        searching_results: The list/dict which store matched patterns.
    """

    def __init__(self, model, dataloader=None, placeholder_shape=None, placeholder_dtype=None):
        """Initialize the attributes."""
        assert isinstance(model, torch.nn.Module)
        if "PyTorchFXModel" in type(model).__name__:
            # neural compressor built-in model type
            self.model = model.model
        else:
            self.model = model
        try:
            self.device = self.model.device
        except:
            self.device = next(self.model.parameters()).device
        # use torch.jit to generate static graph
        self.dataloader = dataloader  # user can set a dataloader to help trace static graph
        self.placeholder_shape = placeholder_shape  # dummy input's shape
        self.placeholder_dtype = placeholder_dtype  # dummy input's data
        self.static_graph = None
        self.flatten_static_graph = None
        self.analyze_dummy_input()
        self.generate_static_graph()
        # save the searching results
        self.target_layers = ["linear"]
        self.search_results = []

    def analyze_dummy_input(self):
        """Analyze the model's input type.

        If no dataloader is specified, searcher will automatically generate a dummy input to
        obtain static graph.
        """
        # if the user already set the dummy inputs, no need to analyze the model
        if self.placeholder_dtype is not None:
            return
        # analyze the model automatically
        first_parameter = None
        for n, p in self.model.named_parameters():
            if first_parameter is not None:
                break
            else:
                first_parameter = p
        if len(first_parameter.shape) == 4:
            # conv op, indicating that this is a cv model
            self.placeholder_shape = [1, 3, 512, 512]
            self.placeholder_dtype = torch.float32
        elif len(first_parameter.shape) == 2:
            # linear or embedding ops, indicating that this is a nlp model
            self.placeholder_shape = [1, 16]
            self.placeholder_dtype = torch.int64
        else:
            logger.warning("Cannot generate dummy input automatically.")
            self.placeholder_shape = [1, 16]
            self.placeholder_dtype = torch.int64
        return

    def generate_static_graph_with_dummyinput(self):
        """Generate static graph from a dummy input, if no dataloader is specified."""
        # static graph generation relies on shape
        dummy_inputs = self.generate_dummy_inputs()
        dummy_inputs = [] + [dummy_inputs]
        if type(self.model).__name__ == "WhisperForConditionalGeneration":
            dummy_inputs = [torch.ones([1, 80, 3000]), torch.ones([1, 448], dtype=torch.int64)]
        logger.info("Generating static graph from original model using auto dummy input: start.")
        try:
            self.static_graph = torch.jit.trace(self.model, dummy_inputs, strict=False)
            # re-org from original static codes.
            self.flatten_static_graph = [l.strip() for l in self.static_graph.inlined_graph.__str__().split("\n")]
            logger.info("Generating static graph from original model using auto dummy input: success.")
        except:
            logger.info("Generating static graph from original model using auto dummy input: failed.")

    def generate_static_graph_with_dataloader(self):
        """Generate static graph from a external dataloader."""
        # dummy_input = self.dataloader[0]
        logger.info("Generating static graph from original model using external data: start.")
        for dummy_input in self.dataloader:
            if isinstance(dummy_input, dict):
                try:
                    dummy_input = dummy_input["input_ids"]
                    self.static_graph = torch.jit.trace(self.model, dummy_input.to(self.device), strict=False)
                except:
                    pass
            else:
                try:
                    for idx in range(len(dummy_input)):
                        dummy_input[idx] = dummy_input[idx].to(self.device)
                    self.static_graph = torch.jit.trace(self.model, dummy_input, strict=False)
                except:
                    try:
                        dummy_input = dummy_input[0]
                        self.static_graph = torch.jit.trace(self.model, dummy_input.to(self.device), strict=False)
                    except:
                        pass
            if self.static_graph is not None:
                # if jit graph is successfully generated, end iteration
                break
        try:
            self.flatten_static_graph = [l.strip() for l in self.static_graph.inlined_graph.__str__().split("\n")]
            logger.info("Generating static graph from original model using external data: success.")
        except:
            logger.warning("Generating static graph from original model using external data: failed.")

    def generate_static_graph(self):
        """Generate static graph with two methods: using dataloader or dummy input."""
        # first do the jit trace using dataloader
        if self.dataloader is not None:
            self.generate_static_graph_with_dataloader()
        # if dataloader based jit trace cannot work or not chosen, use dummy input
        if self.static_graph is not None:
            return
        else:
            self.generate_static_graph_with_dummyinput()

    def generate_dummy_inputs(self):
        """Generate dummy inputs for the model's static graph.

        Return:
            A torch.Tensor passed into the model to generate static graph.
        """
        return torch.ones(self.placeholder_shape, dtype=self.placeholder_dtype).to(self.device)

    def filter_static_code(self, list_in, kw):
        """Obtain sub-list which contains some key words.

        Args:
            list_in: list.
            kw: string.

        Return: a sub-list of list_in, whose members contains kw.
        """
        list_out = []
        for info in list_in:
            if kw in info:
                list_out.append(info)
        return list_out

    def refine_strings(self, string_list):
        """Remove space and tabs in strings."""
        return [s.strip() for s in string_list]

    def analyze_jit_code(self, code):
        """Analyzes and extracts static graph str style's critical information.

        Args:
            code: a str presenting static graph forwarding code

        Return:
            A dict:
                {
                    output_name: "the output node name for this op",
                    input_name: "the input node name for this op",
                    op_type: "the aten::op name",
                    op_trace: "the absolute dir to get this model, in torch.nn.Module's attribute style."
                }
        """

        def remove_weight_or_bias_getattr_op(input_name):
            # %weight and %bias are not related to graph search, therefore skip
            return "%weight" not in input_name and "bias" not in input_name

        # step1 : find outputs' name
        output_names = code.split(":")[0].strip().split(",")
        output_names = self.refine_strings(output_names)
        # step2: find inputs' name
        # use pattern match to find aten::op which includes inputs' name
        aten_pattern = re.compile("aten::.*,")
        aten_regex = aten_pattern.search(code)[0]
        input_pattern = re.compile("\(.*\)")
        input_names = input_pattern.search(aten_regex)[0][1:-1].split(",")
        input_names = filter(remove_weight_or_bias_getattr_op, input_names)
        input_names = self.refine_strings(input_names)
        # step3: obtain the tensor shape of ops
        shape_pattern = re.compile("Float\(.* strides")
        try:
            op_shape = shape_pattern.search(code)[0][6:-9]
        except:
            op_shape = None
        # step4: find the op name (linear, or a act type)
        aten_op_pattern = re.compile("aten::.*\(")
        op_type = aten_op_pattern.search(code)[0][6:-1]
        # step5: find the attribute calling code
        op_trace_pattern = re.compile("scope\:.*\#")
        op_trace = self.get_layer_path_from_jit_code(op_trace_pattern.search(code)[0])
        # step6: compile all information in a dict and return
        res = {
            "output_names": output_names,  # should be a list
            "input_names": input_names,  # should be a list
            "op_shape": op_shape,
            "op_type": op_type,
            "op_trace": op_trace,
        }
        return res

    def search(self):
        """Operations called for entire searching process."""
        raise NotImplementedError

    def get_layer_object_from_jit_codes(self, scope_code):
        """Obtain the specific layer from jit code.

        In jit, scope keyword is a item which use to trace a layer from a model
        For example, for a intermediate layer in Huggingface bert-base, its scope is like:
        scope: __module.bert/__module.bert.encoder/__module.bert.encoder.layer.0/
               __module.bert.encoder.layer.0.intermediate/__module.bert.encoder.layer.0.intermediate.dense #
        example: '__module.bert.encoder.layer.11.intermediate.intermediate_act_fn'

        Args:
            scope_code: a string representing a operator's forward code.

        Return:
            a torch.nn.module: the layer/operator corresponding with scope_code.
        """
        scope_regex = re.compile("scope\: .* \#")
        try:
            scope_part = scope_regex.search(scope_code)[0]
        except:
            logger.warning(f"{scope_code} does contain wanted scope info.")
            return ""
        # strip scope keyword, only keep concrete items
        scope_part = scope_part[7:-2].strip()
        # the last content contains the complete route from top to down
        scope_contents = scope_part.split("/")[-1]
        attrs = scope_contents.split(".")[1:]
        sub_module = self.model
        # iteratively locate the target layer from top(model) to down(layer)
        for attr in attrs:
            sub_module = getattr(sub_module, attr)
        return sub_module

    def get_layer_path_from_jit_code(self, scope_code):
        """Get the module name from its static graph scope code."""
        scope_regex = re.compile("scope\: .* \#")
        try:
            scope_part = scope_regex.search(scope_code)[0]
        except:
            logger.warning(f"{scope_code} does contain wanted scope info.")
            return ""
        # strip scope keyword, only keep concrete items
        scope_part = scope_part[7:-2].strip()
        scope_contents = scope_part.split("/")[-1]
        level_names = scope_contents.split(".")
        level_names_main = ".".join(level_names[1:])
        return level_names_main


class Linear2LinearSearcher(JitBasicSearcher):
    """Static graph searcher for consecutive linear layers.

    Use the static graph to detect some special pattern in a module, there is no need for user to define layer name.
    Automatically search linear layers which can be optimized.

    Args:
        model (torch.nn.Module): The PyTorch model for searching.

    Attributes:
        model: The PyTorch model for searching.
        device: The model's current device type.
        static_graph: The static graph of original model.
        flatten_static_graph: A list of string with the model's static graph inference details.
        target_layers: The layer types the searcher will extract.
        searching_results: The list/dict which store matched patterns.
        target_op_lut: a lookup table for target operators and their corresponding jit codes.
        current_pattern: a searching path to store searching status.
    """

    def __init__(self, model, dataloader=None, placeholder_shape=None, placeholder_dtype=None):
        """Initialize."""
        assert isinstance(model, torch.nn.Module)
        super(Linear2LinearSearcher, self).__init__(model, dataloader, placeholder_shape, placeholder_dtype)
        self.target_op_lut = {}
        self.current_pattern = []
        # initialize target_op_lut
        for op in JIT_SUPPORT_OPS:
            self.target_op_lut[op] = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::" + op)

    def search_frontier_ops_from_node(self, node_name):
        """Search the frontier nodes from a original op's input nodes.

        Args:
            node_name: a node string (%input xxx, %yyy, etc.)

        Return:
            a list of ops, whose output is node_name.
        """
        target_frontier_ops = []
        for op_type, op_codes in self.target_op_lut.items():
            for op_code in op_codes:
                output_names = JitBasicSearcher.analyze_jit_code(self, op_code)["output_names"]
                if output_names.__len__() == 1 and node_name == output_names[0]:
                    target_frontier_ops.append(op_code)
                else:
                    continue
        return target_frontier_ops

    def search_from_root_linear(self, linear_code):
        """Search frontier linears from a linear op."""
        self.current_pattern.clear()
        linear_info = JitBasicSearcher.analyze_jit_code(self, linear_code)
        root_linear_trace = linear_info["op_trace"]
        # data structure to save the results
        results = {
            "root_linear": root_linear_trace,
            "target_frontier_linears": [],
        }

        # start dfs
        def dfs(root_op_code):
            """A dfs step code."""
            op_info = JitBasicSearcher.analyze_jit_code(self, root_op_code)
            op_inputs = op_info["input_names"]
            for op_input in op_inputs:
                frontier_ops = self.search_frontier_ops_from_node(op_input)
                # retrively search the ops
                for frontier_op in frontier_ops:
                    frontier_op_info = JitBasicSearcher.analyze_jit_code(self, frontier_op)
                    if frontier_op_info["op_type"] == "linear":
                        results["target_frontier_linears"].append(frontier_op_info["op_trace"])
                    else:
                        dfs(frontier_op)

        dfs(linear_code)
        return results

    def search(self):
        """Operations called for entire searching process.

        some example:
        A    X    Y
        |     \  /
        B       Z
        A, B, X, Y, Z are all linear layers, some ops including add, mul, dropout can be ignored.
        When we prune B or Z, we can also prune A or X & Y of same channel indices.
        Return:
            A list [
                {
                    "root_linear": str (B or Z),
                    "target_frontier_linears": [str] ([A] or [X, Y])
                }
            ]
        """
        all_linear_structure_results = []
        for linear_code in self.target_op_lut["linear"]:
            search_res = self.search_from_root_linear(linear_code)
            if search_res["target_frontier_linears"].__len__() > 0:
                all_linear_structure_results.append(search_res)
        # Summary
        print_iterables(all_linear_structure_results)
        logger.info(f"Found {all_linear_structure_results.__len__()} linear2linear structures")
        if all_linear_structure_results.__len__() == 0:
            logger.warning("No linear2linear modules are hooked.")
        return all_linear_structure_results

    def from_layer_name_to_object(self, l2l_search_layers):
        """Obtain the layer objects themselves from their names.
        {
            'root_linear': str(attribute),
            'target_frontier_linears': list(str)
        }
        ->
        {
            'root_linear': torch.nn.Linear,
            'target_frontier_linears': [torch.nn.Linear, torch.nn.Linear, ...]
        }
        """
        layer_objs = []
        for item in l2l_search_layers:
            layer_obj = {
                "root_linear": None,
                "target_frontier_linears": [],
            }
            layer_obj["root_linear"] = get_attributes(self.model, item["root_linear"])
            layer_obj["target_frontier_linears"] = [
                get_attributes(self.model, linfo) for linfo in item["target_frontier_linears"]
            ]
            layer_objs.append(layer_obj)
        return layer_objs


class SelfMHASearcher(JitBasicSearcher):
    """Static graph searcher for multi-head attention modules.

    Use the static graph to detect some special pattern in a module, there is no need for user to define layer name.
    Automatically search multi-head attention modules which can be optimized.

    Args:
        model (torch.nn.Module): The PyTorch model for searching.

    Attributes:
        model: The PyTorch model for searching.
        device: The model's current device type.
        static_graph: The static graph of original model.
        flatten_static_graph: A list of string with the model's static graph inference details.
    """

    def __init__(self, model, dataloader=None, placeholder_shape=None, placeholder_dtype=None):
        """Initialize."""
        assert isinstance(model, torch.nn.Module)
        super(SelfMHASearcher, self).__init__(model, dataloader, placeholder_shape, placeholder_dtype)

    def get_head_pattern(self):
        """Obtain head block sizes."""
        hidden_size = self.model.config.hidden_size
        head_size = self.model.config.hidden_size // self.model.config.num_attention_heads
        qkv_pattern = str(head_size) + "xchannel"
        ffn_pattern = "channelx" + str(head_size)
        return qkv_pattern, ffn_pattern

    def gather_mha_inputs(self):
        """Search the multi-head attention modules' query, key, as well as value layers."""
        linears = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::linear")
        linear_infos = [JitBasicSearcher.analyze_jit_code(self, li) for li in linears]
        # generate all nodes' name and their related input node names
        input_counts = {}
        # get all linear modules
        for linfo in linear_infos:
            for input_name in linfo["input_names"]:
                if linfo["op_type"] == "linear" and input_name in input_counts:
                    input_counts[input_name] += 1
                elif linfo["op_type"] == "linear" and input_name not in input_counts:
                    input_counts[input_name] = 1
                else:
                    # op which is not linear, skip
                    continue
        input_counts_filtered = {}
        # in our strategy, when three linear layers share the same input, they should be query, key, and value
        for k, v in input_counts.items():
            if v >= 3:
                # attention's number
                input_counts_filtered[k] = v
            else:
                continue
        return input_counts_filtered

    def gather_linear_from_input(self, input_names: dict):
        """Gather query, key and value layers of the same self-attention module together."""
        linear_clusters = {}
        linears = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::linear")
        for li in linears:
            linfo = JitBasicSearcher.analyze_jit_code(self, li)
            for input_name in linfo["input_names"]:
                if input_name in input_names:
                    if input_name in linear_clusters:
                        linear_clusters[input_name].append(li)
                    else:
                        linear_clusters[input_name] = [li]
                else:
                    continue
        return linear_clusters

    def extract_qkv_from_linears(self, linears):
        """Extract linear cluster with same inputs, same op shape, and size is 3 (qkv)

        Args:
            linears: A dict, key is input name, value is a list of linear layers jit code

        Return:
            A dict contains only qkv linear layers.
        """
        qkv_clusters = {}
        for input_name, input_linked_linears in linears.items():
            linfos = [JitBasicSearcher.analyze_jit_code(self, il) for il in input_linked_linears]
            # step 1: statistics of linears clusters with same shape.
            op_shape_lut = {}
            for linfo in linfos:
                op_shape = linfo["op_shape"]
                if op_shape_lut.get(op_shape, None) is None:
                    op_shape_lut[op_shape] = 1
                else:
                    op_shape_lut[op_shape] += 1
            qkv_related_op_shape = []
            for op_shape_lut_key in op_shape_lut:
                if op_shape_lut[op_shape_lut_key] == 3:
                    qkv_related_op_shape.append(op_shape_lut_key)
                else:
                    continue
            # step 2: extract qkv layers
            qkv_linears = []
            for linfo in linfos:
                if linfo["op_shape"] in qkv_related_op_shape:
                    qkv_linears.append(linfo["op_trace"])
                else:
                    continue
            qkv_clusters[input_name] = qkv_linears
        qkv_clusters_filtered = {}
        for key in qkv_clusters.keys():
            if qkv_clusters[key].__len__() == 3:
                qkv_clusters_filtered[key] = qkv_clusters[key][:]
        return qkv_clusters_filtered

    def search_ffn_from_qkv(self, qkv_clusters):
        """Search the related ffn linear module related to every self-attention."""
        linear_lut = []
        for n, m in self.model.named_modules():
            if type(m).__name__ == "Linear":
                linear_lut.append(n)
        # initialize the qkv data structure
        self_attn_list = []
        for input_name in qkv_clusters:
            self_attn = {"qkv": qkv_clusters[input_name][:], "ffn": []}
            for idx in range(len(linear_lut)):
                if idx >= 1 and (linear_lut[idx - 1] in self_attn["qkv"]) and (linear_lut[idx] not in self_attn["qkv"]):
                    # this means we find the first linear layer after qkv
                    self_attn["ffn"].append(linear_lut[idx])
                    break
                else:
                    continue
            self_attn_list.append(self_attn)
            del self_attn
        return self_attn_list

    def search(self, split_qkv_ffn=True):
        """Operations called for entire searching process.

        Args:
            split_qkv_ffn: a bool. Whether to rearrange searched attention heads' linear layers.
                if True: return two lists: one contains all query, key and value layers,
                    the other contains all forward layers.
                if False: only return one list containing self-attention's linear layers,
                    query, key, value layers and forward layers are not split.

        Return:
            two lists containing self-attention modules' layer names.
        """
        input_names_for_linears = self.gather_mha_inputs()
        linear_clusters = self.gather_linear_from_input(input_names_for_linears)
        qkv_clusters = self.extract_qkv_from_linears(linear_clusters)
        self_attn_list = self.search_ffn_from_qkv(qkv_clusters)
        # summary
        print_iterables(self_attn_list)
        logger.info(f"Found {self_attn_list.__len__()} MHA modules")
        if self_attn_list.__len__() == 0:
            logger.warning("No MHA modules are hooked.")
        if not split_qkv_ffn:
            return self_attn_list, None
        else:
            # put all qkv into one list, all ffn into another list
            qkv_list = []
            ffn_list = []
            for item in self_attn_list:
                qkv_list += item["qkv"]
                ffn_list += item["ffn"]
            return qkv_list, ffn_list

    def from_layer_name_to_object(self, mha_search_layers):
        """Obtain the layer object themselves from their names.
        [
            {
                'qkv': ['query_layer_name', 'key_layer_name', 'value_layer_name'],
                'ffn': ['attention_ffn_name']
                'mha_name': ['mha_name'] # bert.encoder.layer.0, etc.
                'mha_module': [torch.nn.Module] # which corresponds to mha_name above.
            }
            ...
        ]
        ->
        [
            {
                'qkv_name': ['query_layer_name', 'key_layer_name', 'value_layer_name'],
                'ffn_name': ['attention_ffn_name'],
                'mha_name': ['mha_name'] (keep not change),
                'qkv_module': [torch.nn.Linear, torch.nn.Linear, torch.nn.Linear],
                'ffn_module': [torch.nn.Linear],
                'mha_module': [torch.nn.Module] (keep not change),
            }
            ...
        ]
        """
        layer_objs = []
        for mha_search_layer in mha_search_layers:
            # copy layer names
            layer_obj = {
                "qkv_name": mha_search_layer["qkv"][:],
                "ffn_name": mha_search_layer["ffn"][:],
                "mha_name": mha_search_layer["mha_name"][:],
            }
            # obtain pytorch module
            layer_obj["qkv_module"] = [get_attributes(self.model, layer_name) for layer_name in mha_search_layer["qkv"]]
            layer_obj["ffn_module"] = [get_attributes(self.model, layer_name) for layer_name in mha_search_layer["ffn"]]
            # we can directly copy since we have already obtained this module before
            layer_obj["mha_module"] = mha_search_layer["mha_module"][:]
            layer_objs.append(layer_obj)
        return layer_objs

    def obtain_mha_module(self, self_attention_list):
        """Return the attention module object (qkv & ffn's common module).

        self_attention_list
        [
            {
                'qkv': ['query_layer_name', 'key_layer_name', 'value_layer_name'],
                'ffn': ['attention_ffn_name']
            }
            ...
        ]
        ->
        [
            {
                'qkv': ['query_layer_name', 'key_layer_name', 'value_layer_name'],
                'ffn': ['attention_ffn_name'],
                'mha_name': ['mha_name'],
                'mha_module': [torch.nn.Module]
            }
            ...
        ]
        """
        for idx in range(len(self_attention_list)):
            # get query layer name
            # get attn_output layer name
            qkv_layer_name = self_attention_list[idx]["qkv"]
            ffn_layer_name = self_attention_list[idx]["ffn"]
            # problematic implementations
            # mha_module_name = get_common_module(qkv_layer_name, ffn_layer_name)
            mha_module_name = get_common_module(qkv_layer_name[0], qkv_layer_name[-1])
            self_attention_list[idx]["mha_name"] = [mha_module_name]
            self_attention_list[idx]["mha_module"] = [
                get_attributes(self.model, mha_module_name) for mha_module_name in self_attention_list[idx]["mha_name"]
            ]
        return self_attention_list


class ClassifierHeadSearcher(object):
    """Static graph searcher for multi-head attention modules.

    Use the static graph to detect final classifier head in a module, there is no need for user to define layer name.
    Automatically search multi-head attention modules which can be optimized.

    Args:
        model (torch.nn.Module): The PyTorch model for searching.

    Attributes:
        model: The PyTorch model for searching.
        device: The model's current device type.
        static_graph: The static graph of original model.
        flatten_static_graph: A list of string with the model's static graph inference details.
    """

    def __init__(self, model):
        """Initialize."""
        assert isinstance(model, torch.nn.Module)
        super(ClassifierHeadSearcher, self).__init__()
        self.model = model
        self.pruning_ops = ["Linear", "Conv2d"]
        self.excluded_ops = ["Dropout"]  # to be extended

    def search(self, return_name=True):
        all_modules = []
        all_lc_modules = []
        for n, m in self.model.named_modules():
            if type(m).__name__ not in self.excluded_ops:
                all_modules.append(n)
                if type(m).__name__ in self.pruning_ops:
                    all_lc_modules.append(n)
            else:
                continue
        last_lc = all_lc_modules[-1]
        if last_lc == all_modules[-1]:
            return last_lc
        else:
            return None


class ClassifierHeadSearcherTF(object):
    """Static graph searcher for multi-head attention modules.

    Use the static graph to detect final classifier head in a module, there is no need for user to define layer name.
    Automatically search multi-head attention modules which can be optimized.

    Args:
        model (tf.keras.Model): The Keras model for searching.

    Attributes:
        model: The Keras model for searching.
        device: The model's current device type.
        static_graph: The static graph of original model.
        flatten_static_graph: A list of string with the model's static graph inference details.
    """

    def __init__(self, model):
        """Initialize."""
        assert isinstance(model, tf.keras.Model)
        super(ClassifierHeadSearcherTF, self).__init__()
        self.model = model
        self.pruning_ops = ["Dense", "Conv2d"]
        self.excluded_ops = ["Dropout"]  # to be extended

    def search(self, return_name=True):
        all_modules = []
        all_lc_modules = []
        for layer in self.model.layers:
            if layer.__class__.__name__ not in self.excluded_ops:
                all_modules.append(layer.name)
                if layer.__class__.__name__ in self.pruning_ops:
                    all_lc_modules.append(layer.name)
            else:
                continue
        last_lc = all_lc_modules[-1]
        if last_lc == all_modules[-1]:
            return last_lc
        return None
