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

import torch
import re
from ..utils import logger

JIT_SUPPORT_OPS = ['linear', 'gelu', 'mul'] # linear and all act_fn supported by pytorch-aten extension

# MHA_SUPPORT_NAMES = ["q", "k", "v"]

def get_attributes(module: torch.nn.Module, attrs: str):
    """Get a multi-level descent module of module.

    Args:
        module (torch.nn.Module): The torch module.
        attrs (str): The attributes' calling path.
        
    Returns:
        attr: The target attribute of the module.
    """
    attrs_list = attrs.split('.')
    sub_module = module
    while attrs_list:
        attr = attrs_list.pop(0)
        sub_module = getattr(sub_module, attr)
    return sub_module

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
        recipe (dict): A dict containing infomation of the searching pattern.
            
    Attributes:
        model: The PyTorch model for searching.
        recipe: A dict containing infomation of the searching pattern.
        targets: The basic module's name which contains searching pattern.
        searching_results: The list/dict which store matched patterns.
    """

    def __init__(self, model: torch.nn.Module, recipe: dict):
        """Initialize the attributes."""
        if "PyTorchFXModel" in type(model).__name__:
            # neural compressor build-in model type
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

    def __init__(self, model):
        """Initialize the attributes."""
        if "PyTorchFXModel" in type(model).__name__:
            # neural compressor build-in model type
            self.model = model.model
        else:
            self.model = model
        self.device = self.model.device
        # use torch.jit to generate static graph
        self.static_graph = None
        self.flatten_static_graph = None
        self.generate_static_graph()
        # save the searching results
        self.target_layers = ['linear']
        self.search_results = []

    def generate_static_graph(self):
        """Operations called when generate the model's static graph."""
        logger.info(f"Generating jit tracing from original model.")
        dummy_inputs = self.generate_dummy_inputs()
        self.static_graph = torch.jit.trace(self.model, dummy_inputs, strict=False)
         # re-org from original static codes. 
        self.flatten_static_graph = [l.strip() for l in self.static_graph.inlined_graph.__str__().split('\n')]
    
    def generate_dummy_inputs(self, shape=[1, 16], dtype=torch.int64):
        """Generate dummy inputs for the model's static graph.

        Args:
            shape: the dummy input's shape.
            dtype: the dummy input's date type. For nlp tasks, it should be torch.int64 (long).
        
        Return:
            A torch.Tensor.
        """
        return torch.ones(shape, dtype=dtype).to(self.device)

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

    def analyze_code(self, code):
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
        # find output's name
        output_name = code.split(":")[0].strip()
        # use pattern match to find aten::op and input's name
        aten_pattern = re.compile('aten::.*,')
        aten_regex = aten_pattern.search(code)[0]
        input_pattern = re.compile("\(.*\)")
        # only obtain the first input name, for linear, act_fn, the first input name refer to hidden state
        input_name = input_pattern.search(aten_regex)[0][1:-1].split(",")[0]
        # obtain the op name (linear, or a act type)
        aten_op_pattern = re.compile('aten::.*\(')
        op_type = aten_op_pattern.search(code)[0][6:-1]
        # obtain the op_trace
        op_trace_pattern = re.compile('scope\:.*\#')
        op_trace = op_trace_pattern.search(code)[0]
        res = {
            "output_name": output_name,
            "input_name": input_name,
            "op_type": op_type,
            "op_trace": op_trace,
        }
        return res

    def search(self):
        """Operations called for entire searching process."""
        raise NotImplementedError

    def get_layer_for_all(self):
        """Extract target layers from matched patterns.

        After searching process, target patterns are stored in self.search_results.
        This function obtains obtains the layer object (torch.nn.Module) from self.search_results.
        By default, self.target_layer is ['Linear'], therefore this function only obtain linear layers,
        and store them in self.search_results.
        """
        results = []
        for pattern in self.search_results:
            pattern_layer = []
            for layer in pattern:
                if layer['op_type'] in self.target_layers: 
                    pattern_layer.append(self.get_layers(layer['op_trace']))
                else:
                    continue
            results.append(pattern_layer)
        self.search_results.clear()
        self.search_results += results

    def get_layers(self, scope_code):
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
        scope_regex = re.compile('scope\: .* \#')
        try:
            scope_part = scope_regex.search(scope_code)[0]
        except:
            logger.warning(f"{scope_code} does contain wanted scope info.")
            return ""
        # strip scope keyword, only keep contrete items
        scope_part = scope_part[7:-2].strip()
        # the last content contains the complete route from top to down
        scope_contents = scope_part.split('/')[-1]
        attrs = scope_contents.split('.')[1:]
        sub_module = self.model
        # iteratively locate the target layer from top(model) to down(layer)
        for attr in attrs:
            sub_module = getattr(sub_module, attr)
        return sub_module
    
    def get_layer_name(self, scope_code):
        """
        Get the module name from its static graph scope code.
        """
        scope_regex = re.compile('scope\: .* \#')
        try:
            scope_part = scope_regex.search(scope_code)[0]
        except:
            logger.warning(f"{scope_code} does contain wanted scope info.")
            return ""
        # strip scope keyword, only keep contrete items
        scope_part = scope_part[7:-2].strip()
        scope_contents = scope_part.split('/')[-1]
        level_names = scope_contents.split('.')
        level_names_main =  ".".join(level_names[1:])
        return level_names_main

class PathSearcher(JitBasicSearcher):
    """Static graph searcher.

    Use the static graph to detect some special pattern in a module, there is no need for user to define layer name.
    Need to provide a string to indicate the structure/sequence of the target pattern.
    
    Args:
        model (torch.nn.Module): The PyTorch model for searching.
        target_pattern (str): a string presenting the pattern to search (use '/' to link different layer names.)
            
    Attributes:
        model: The PyTorch model for searching.
        device: The model's current device type.
        static_graph: The static graph of original model.
        flatten_static_graph: A list of string with the model's static graph inference details.
        target_layers: The layer types the searcher will extract.
        searching_results: The list/dict which store matched patterns.
        target_path: a string presenting the pattern to search (use '/' to link different layer names.)
        target_op: a set containing target operators' names.
        target_op_lut: a lookup table for target operators and their corresponding jit codes.
        current_pattern: a searching path to store searching status.
    """

    def __init__(self, model, target_pattern='linear/gelu/linear'):
        """Initialize."""
        super(PathSearcher, self).__init__(model)
        # some search related attribuites
        self.target_pattern = target_pattern
        # re-org target_pattern to obtain target ops
        self.target_path = self.target_pattern.split('/')
        self.target_ops = set(self.target_path)
        self.target_op_lut = {}
        self.current_pattern = []
    
    def search(self):
        """Operations called for entire searching process."""
        # step 1: establish search space within all interested ops, saved in self.target_op_lut
        self.search_results.clear()
        self.target_op_lut.clear()
        self.current_pattern.clear()
        for op in self.target_ops:
            self.target_op_lut[op] = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::"+op)
        def dfs():
            """Operations called for one single search step."""
            # if target pattern is found
            if len(self.current_pattern) == len(self.target_path):
                # find the target pattern
                self.search_results.append(self.current_pattern[:])
                return
            # else continue searching
            next_op_type = self.target_path[len(self.current_pattern)]
            for op_code in self.target_op_lut[next_op_type]:
                op_info = JitBasicSearcher.analyze_code(self, op_code)
                if len(self.current_pattern) == 0: 
                    self.current_pattern.append(op_info)
                    dfs()
                    self.current_pattern.pop()
                else:
                    if op_info['input_name'] == self.current_pattern[-1]['output_name']:
                        self.current_pattern.append(op_info)
                        dfs()
                        self.current_pattern.pop()
                    else:
                        continue 
        # step 2: dfs  
        # execute dfs-based pattern matching
        dfs()
        logger.info(f"Found {len(self.search_results)} pattern {self.target_pattern} in {type(self.model).__name__}")
        JitBasicSearcher.get_layer_for_all(self)
        return self.search_results

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

    def __init__(self, model):
        """Initialize."""
        super(Linear2LinearSearcher, self).__init__(model)
        self.target_op_lut = {}
        self.current_pattern = []

    def search(self, return_name = False):
        """Operations called for entire searching process."""
        self.search_results.clear()
        self.target_op_lut.clear()
        self.current_pattern.clear()
        for op in JIT_SUPPORT_OPS:
            self.target_op_lut[op] = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::"+op)

        def dfs():
            """Operations called for one single search step."""
            # ends up with another linear layer, successfully obtain a linear2linear pattern
            if len(self.current_pattern) > 1 and self.current_pattern[-1]['op_type'] == "linear":
                self.search_results.append(self.current_pattern[:])
                return
            # continue searching
            lastest_ops = self.current_pattern[-1]
            lastest_ops_outputs = lastest_ops['output_name']
            for next_op_type, next_op_codes in self.target_op_lut.items():
                for next_op_code in next_op_codes:
                    next_op_info = JitBasicSearcher.analyze_code(self, next_op_code)
                    next_op_info_input = next_op_info['input_name']
                    if next_op_info_input == lastest_ops_outputs:
                        self.current_pattern.append(next_op_info)
                        dfs()
                        self.current_pattern.pop()
                    else:
                        continue

        for init_op_code in self.target_op_lut['linear']:
            init_op_info = JitBasicSearcher.analyze_code(self, init_op_code)
            self.current_pattern.append(init_op_info)
            dfs()
            self.current_pattern.pop()

        logger.info(f"Found {len(self.search_results)} target pattern 'linear2linear' in {type(self.model).__name__}")
        if not return_name: 
            # return the module object instead of module name
            JitBasicSearcher.get_layer_for_all(self)
            return self.search_results
        else:
            name_results = []
            for item in self.search_results:
                name_item = [JitBasicSearcher.get_layer_name(self, layer_info['op_trace']) for layer_info in item]
                name_results.append(name_item)
            return name_results


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

    def __init__(self, model):
        """Initialize."""
        super(SelfMHASearcher, self).__init__(model)

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
        linear_infos = [JitBasicSearcher.analyze_code(self, li) for li in linears]
        # generate all nodes' name and their related input node names
        input_counts = {}
        # get all linear modules
        for linfo in linear_infos:
            if linfo['input_name'] in input_counts:
                input_counts[linfo['input_name']] += 1
            else:
                input_counts[linfo['input_name']] = 1
        input_counts_filtered = {}
        # in our strategy, when three linear layers share the same input, they should be query, key, and value
        for k, v in input_counts.items():
            if v == 3:
                # attention's number
                input_counts_filtered[k] = v
            else: 
                continue
        return input_counts_filtered
    
    def gather_qkv_from_input(self, input_names: dict):
        """Gather query, key and value layers of the same self-attention module together."""
        qkv_clusters = {}
        linears = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::linear")
        for li in linears:
            linfo = JitBasicSearcher.analyze_code(self, li)
            if linfo['input_name'] in input_names:
                if linfo['input_name'] in qkv_clusters:
                    qkv_clusters[linfo['input_name']].append(linfo['op_trace'])
                else:
                    qkv_clusters[linfo['input_name']] = [linfo['op_trace']]
            else:
                continue
        return qkv_clusters
    
    def search_ffn_from_qkv(self, qkv_clusters):
        """Search the related ffn linear module related to every self-attention."""
        linear_lut = []
        for n, m in self.model.named_modules():
            if type(m).__name__ == "Linear":
                linear_lut.append(n)
        # initialize the qkv data structure 
        self_attn_list = []
        for input_name in qkv_clusters:
            self_attn = {
                "qkv": qkv_clusters[input_name][:],
                "ffn": []
            }
            for idx in range(len(linear_lut)):
                if idx >= 1 and (linear_lut[idx-1] in self_attn["qkv"]) and (linear_lut[idx] not in self_attn["qkv"]):
                    # this means we find the first linear layer after qkv
                    self_attn["ffn"].append(linear_lut[idx])
                    break
                else:
                    continue
            self_attn_list.append(self_attn)
            del self_attn
        return self_attn_list

    def search(self, split_qkv_ffn = True):
        """Operations called for entire searching process.

        Args:
            split_qkv_ffn: a bool. Whether to rearrange searched attention heads' linear layers.
                if True: return two lists: one contains all query, key and value layers, 
                    the other contains all forward layers.
                if False: only return one list containing self-attention's linear layers, 
                    query, key, value layers and forward layers are not splited. 
        
        Return:
            two lists containing self-attention modules' layer names.

        """
        input_names_for_linears = self.gather_mha_inputs()
        qkv_clusters = self.gather_qkv_from_input(input_names_for_linears)
        qkv_clusters_main = {}
        for input_name in qkv_clusters:
            qkv_clusters_main[input_name] = [
            JitBasicSearcher.get_layer_name(self, scope_code) for scope_code in qkv_clusters[input_name]
        ]
        self_attn_list = self.search_ffn_from_qkv(qkv_clusters_main)
        if not split_qkv_ffn:
            return self_attn_list, None
        else:
            # put all qkv into one list, all ffn into another list
            qkv_list = []
            ffn_list = []
            for item in self_attn_list:
                qkv_list += item["qkv"]
                ffn_list += item['ffn']
            return qkv_list, ffn_list
