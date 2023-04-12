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

JIT_SUPPORT_OPS = ['linear', 'gelu', 'silu', 'relu', 'mul', 'add'] # linear and all act_fn supported by pytorch-aten extension

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

    def __init__(self, model, placeholder_shape = None, placeholder_dtype = None):
        """Initialize the attributes."""
        if "PyTorchFXModel" in type(model).__name__:
            # neural compressor build-in model type
            self.model = model.model
        else:
            self.model = model
        try:
            self.device = self.model.device
        except:
            self.device = next(self.model.parameters()).device
        # use torch.jit to generate static graph
        self.placeholder_shape = placeholder_shape # dummy input
        self.placeholder_dtype = placeholder_dtype # dummy input
        self.static_graph = None
        self.flatten_static_graph = None
        self.analyze_dummy_input()
        self.generate_static_graph()
        # save the searching results
        self.target_layers = ['linear']
        self.search_results = []
    
    def analyze_dummy_input(self):
        """Analyze the model's input type."""
        # if the user already set the dummy inputs, no need to analyze the model
        if self.placeholder_dtype != None and self.placeholder_dtype != None:
            return
        # analyze the model automatically
        first_parameter = None
        for n, p in self.model.named_parameters():
            if first_parameter != None:
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
            logger.warning("Cannot generate dummy input automatically, please set it manually when initialzation.")
            self.placeholder_shape = [1, 16]
            self.placeholder_dtype = torch.int64
        return

    def generate_static_graph(self):
        """Operations called when generate the model's static graph."""
        logger.info(f"Generating jit tracing from original model.")
        # static graph generation relies on shape
        dummy_inputs = self.generate_dummy_inputs()
        self.static_graph = torch.jit.trace(self.model, dummy_inputs, strict=False)
         # re-org from original static codes. 
        self.flatten_static_graph = [l.strip() for l in self.static_graph.inlined_graph.__str__().split('\n')]
    
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
            "output_name": output_name, # should be a list
            "input_name": input_name, # shoule be a list
            "op_type": op_type,
            "op_trace": op_trace,
        }
        return res
    
    def refine_strings(self, string_list):
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
        output_names = code.split(":")[0].strip().split(',')
        output_names = self.refine_strings(output_names)
        # step2: find inputs' name
        # use pattern match to find aten::op which includes inputs' name
        aten_pattern = re.compile('aten::.*,')
        aten_regex = aten_pattern.search(code)[0]
        input_pattern = re.compile("\(.*\)")
        input_names = input_pattern.search(aten_regex)[0][1:-1].split(",")
        input_names = filter(remove_weight_or_bias_getattr_op, input_names)
        input_names = self.refine_strings(input_names)
        # step3: find the op name (linear, or a act type)
        aten_op_pattern = re.compile('aten::.*\(')
        op_type = aten_op_pattern.search(code)[0][6:-1]
        # step4: find the 
        op_trace_pattern = re.compile('scope\:.*\#')
        op_trace = self.get_layer_path_from_jit_code(op_trace_pattern.search(code)[0])
        # step5: compile all information in a dict and return
        res = {
            "output_names": output_names, # should be a list
            "input_names": input_names, # shoule be a list
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
                    pattern_layer.append(self.get_layer_object_from_jit_codes(layer['op_trace']))
                else:
                    continue
            results.append(pattern_layer)
        self.search_results.clear()
        self.search_results += results

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
    
    def get_layer_path_from_jit_code(self, scope_code):
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
        # initialize target_op_lut
        for op in JIT_SUPPORT_OPS:
            self.target_op_lut[op] = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::"+op)

    # def search(self, return_name = False):
    #     """Operations called for entire searching process."""
    #     self.search_results.clear()
    #     self.target_op_lut.clear()
    #     self.current_pattern.clear()
    #     for op in JIT_SUPPORT_OPS:
    #         self.target_op_lut[op] = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::"+op)

    #     def dfs():
    #         """Operations called for one single search step."""
    #         # ends up with another linear layer, successfully obtain a linear2linear pattern
    #         if len(self.current_pattern) > 1 and self.current_pattern[-1]['op_type'] == "linear":
    #             self.search_results.append(self.current_pattern[:])
    #             return
    #         # continue searching
    #         lastest_ops = self.current_pattern[-1]
    #         lastest_ops_outputs = lastest_ops['output_name']
    #         for next_op_type, next_op_codes in self.target_op_lut.items():
    #             for next_op_code in next_op_codes:
    #                 next_op_info = JitBasicSearcher.analyze_code(self, next_op_code)
    #                 next_op_info_input = next_op_info['input_name']
    #                 if next_op_info_input == lastest_ops_outputs:
    #                     self.current_pattern.append(next_op_info)
    #                     dfs()
    #                     self.current_pattern.pop()
    #                 else:
    #                     continue

    #     for init_op_code in self.target_op_lut['linear']:
    #         init_op_info = JitBasicSearcher.analyze_code(self, init_op_code)
    #         self.current_pattern.append(init_op_info)
    #         dfs()
    #         self.current_pattern.pop()

    #     logger.info(f"Found {len(self.search_results)} target pattern 'linear2linear' in {type(self.model).__name__}")
    #     if not return_name: 
    #         # return the module object instead of module name
    #         JitBasicSearcher.get_layer_for_all(self)
    #         return self.search_results
    #     else:
    #         name_results = []
    #         for item in self.search_results:
    #             name_item = [JitBasicSearcher.get_layer_path_from_jit_code(self, layer_info['op_trace']) for layer_info in item]
    #             name_results.append(name_item)
    #         return name_results

    def search_frontier_ops_from_node(self, node_name):
        # node_name is a member in output_names and input_names: %xxx for example
        target_frontier_ops = []
        for op_type, op_codes in self.target_op_lut.items():
            for op_code in op_codes:
                output_names = JitBasicSearcher.analyze_jit_code(self, op_code)['output_names']
                if output_names.__len__() == 1 and node_name == output_names[0]:
                    target_frontier_ops.append(op_code)
                else:
                    continue
        return target_frontier_ops

    def search_from_root_linear(self, linear_code):
        # obtain linear tree from one root linear, search from latter to frontier
        self.current_pattern.clear()
        linear_info = JitBasicSearcher.analyze_jit_code(self, linear_code)
        root_linear_trace = linear_info['op_trace']
        # data structure to save the results
        results = {
            "root_linear": root_linear_trace,
            "target_frontier_linears": [],
        }
        # start dfs
        def dfs(root_op_code):
            op_info = JitBasicSearcher.analyze_jit_code(self, root_op_code)
            op_inputs = op_info['input_names']
            for op_input in op_inputs:
                frontier_ops = self.search_frontier_ops_from_node(op_input)
                # retrively search the ops
                for frontier_op in frontier_ops:
                    frontier_op_info = JitBasicSearcher.analyze_jit_code(self, frontier_op)
                    if frontier_op_info['op_type'] == 'linear':
                        results['target_frontier_linears'].append(frontier_op_info['op_trace'])
                    else:
                        dfs(frontier_op)
        dfs(linear_code)
        return results

    def search(self):
        all_linear_structure_results = []
        for linear_code in self.target_op_lut['linear']:
            search_res = self.search_from_root_linear(linear_code)
            if search_res['target_frontier_linears'].__len__() > 0:
                all_linear_structure_results.append(search_res)
        #import pdb;pdb.set_trace()
        # Summary
        for item in all_linear_structure_results:
            logger.info(item)
        logger.info(f"Found {all_linear_structure_results.__len__()} linear2linear structures")
        return all_linear_structure_results
    
    def from_layer_name_to_object(self, l2l_search_layers):
        layer_objs = []
        for item in l2l_search_layers:
            layer_obj = {
                "root_linear": None,
                "target_frontier_linears": [],
            }
            layer_obj['root_linear'] = get_attributes(self.model, item['root_linear'])
            layer_obj['target_frontier_linears'] = [
                get_attributes(self.model, linfo) for linfo in item['target_frontier_linears']
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
            JitBasicSearcher.get_layer_path_from_jit_code(self, scope_code) for scope_code in qkv_clusters[input_name]
        ]
        self_attn_list = self.search_ffn_from_qkv(qkv_clusters_main)
        # summary
        logger.info(f"Found {self_attn_list.__len__()} MHA modules")
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

class ClassifierHeadSearcher(JitBasicSearcher):
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
        super(ClassifierHeadSearcher, self).__init__(model)
        self.pruning_ops = ["Linear", "Conv2d"]
    
    def search(self, return_name=True):
        # import pdb;pdb.set_trace()
        all_modules = []
        all_lc_modules = []
        for n, m in self.model.named_modules():
            all_modules.append(n)
            if type(m).__name__ in self.pruning_ops:
                all_lc_modules.append(n)
        # import pdb;pdb.set_trace()
        last_lc = all_lc_modules[-1]
        if last_lc == all_modules[-1]: return last_lc
        else: return None