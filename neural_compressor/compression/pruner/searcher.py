"""Searcher."""
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

JIT_SUPPORT_OPS = ['linear', 'gelu', 'mul'] # linear and all act_fn supported by pytorch-aten extension

def get_attributes_multi_level(module: torch.nn.Module, attrs: str):
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
    """
    If users exactly know the target layer's name to call, then can define a recipe to locate target layers efficiently. 
    """
    def __init__(self, model: torch.nn.Module, recipe: dict):
        """
        recipe_samples = {
            {'BertLayer': ["intermediate.dense", "output.dense"],}
            'BertAttention': ["self.query", "self.key", "self.value", "output.dense"]
        }
        """
        self.model = model
        self.recipe = recipe
        self.targets = list(self.recipe.keys())
        self.search_results = {}

    def search(self, target_name):
        #assert target_name in STURCTURE_RECIPES, print(f"{target_name} structure is not support for search.")
        self.search_results.clear()
        self.dfs_search(self.model, type(self.model).__name__, target_name)
        return self.search_results
    
    def dfs_search(self, module, module_name, target_name):
        # find a target module
        module_type = type(module).__name__
        if module_type in self.targets:
            sublayers = [get_attributes_multi_level(module, sublayer_name) for sublayer_name in self.recipe[module_type]]
            self.search_results[module_name] = sublayers
        # recursively search
        for n, m in module.named_children():
            self.dfs_search(m, n, target_name)


class JitBasicSearcher(object):
    """
    By converting a PyTorch Model into a static model using torch.jit.trace()/script(),
    we can trace some special pattern in the model and optimizer them automatically.
    This class provide some basic functionality for jit searcher
    Including generating dummy inputs, generating static graph, analyzing static graph.
    """
    def __init__(self, model):
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
        print(f"Generating jit tracing from original model.")
        dummy_inputs = self.generate_dummy_inputs()
        self.static_graph = torch.jit.trace(self.model, dummy_inputs, strict=False)
         # re-org from original static codes. 
        self.flatten_static_graph = [l.strip() for l in self.static_graph.inlined_graph.__str__().split('\n')]
    
    def generate_dummy_inputs(self, shape=[1, 16], dtype=torch.int64):
        """
        shape: list [1, 16], [1, 3, 228, 228] etc.
        to generate a static graph, we need to generate a dummy input for model.
        default: a sequence data for nlp model (Bert, GPTJ, etc.)
        """
        return torch.ones(shape, dtype=dtype).to(self.device)

    def filter_static_code(self, list_in, kw):
        """
        filter out a sublist, whose members contain kw
        """
        list_out = []
        for info in list_in:
            if kw in info:
                list_out.append(info)
        return list_out

    def analyze_code(self, code):
        """
        Analyzes and extracts static graph str style's critical information.
        input: a single-line static graph forwarding code
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
        """
        Core searching function will be implemented in children classes for specific aims. 
        """
        raise NotImplementedError

    def get_layer_for_all(self):
        """
        After searching process, target patterns are stored in self.search_results.
        This function obtains obtains the layer object (torch.nn.Module) from self.search_results
        By default, self.target_layer is ['Linear'], therefore this function only obtain linear layers
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
        """
        # in jit, scope keyword is a item which use to trace a layer's heirachy in a model
        # and it has a tower-like structure. For example, for a intermediate layer in bert-base its scope is like:
        # scope: __module.bert/__module.bert.encoder/__module.bert.encoder.layer.0/__module.bert.encoder.layer.0.intermediate/__module.bert.encoder.layer.0.intermediate.dense #
        # example: '__module.bert.encoder.layer.11.intermediate.intermediate_act_fn'
        """
        scope_regex = re.compile('scope\: .* \#')
        try:
            scope_part = scope_regex.search(scope_code)[0]
        except:
            print(f"{scope_code} does contain wanted scope info.")
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

class PathSearcher(JitBasicSearcher):
    """
    Use jit to detect some special pattern in a module, there is no need for user to define layer name.
    User need to provide a string to indicate the structure of target pattern
    ""
    """
    def __init__(self, model, target_pattern='linear/gelu/linear'):
        super(PathSearcher, self).__init__(model)
        # some search related attribuites
        self.target_pattern = target_pattern
        # re-org target_pattern to obtain target ops
        self.target_path = self.target_pattern.split('/')
        self.target_ops = set(self.target_path)
        self.target_op_lut = {}
        self.current_pattern = []
    
    def search(self):
        """
        By define a target path, we use dfs to search matched patterns
        """
        # step 1: establish search space within all interested ops, saved in self.target_op_lut
        self.search_results.clear()
        self.target_op_lut.clear()
        self.current_pattern.clear()
        for op in self.target_ops:
            self.target_op_lut[op] = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::" + op)
        def dfs():
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
        print(f"Found {len(self.search_results)} target pattern {self.target_pattern} in model {type(self.model).__name__}")
        JitBasicSearcher.get_layer_for_all(self)
        return self.search_results

class Linear2LinearSearcher(JitBasicSearcher):
    """
    A downstream task of PathSearcher, restrict the pattern to be "Linear2Linear":
    Linear2Linear is pattern with start with Linear and ends up with Linear.
    Between two Linears, activation, mul, etc ops are supported
    Due to this restriction, there is no need to provide a specific "target_pattern" comparing to PathSearcher.
    """
    def __init__(self, model):
        super(Linear2LinearSearcher, self).__init__(model)
        self.target_op_lut = {}
        self.current_pattern = []

    def search(self):
        self.search_results.clear()
        self.target_op_lut.clear()
        self.current_pattern.clear()
        for op in JIT_SUPPORT_OPS:
            self.target_op_lut[op] = JitBasicSearcher.filter_static_code(self, self.flatten_static_graph, "aten::" + op)

        def dfs():
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

        print(f"Found {len(self.search_results)} target pattern 'linear2linear' in model {type(self.model).__name__}")
        JitBasicSearcher.get_layer_for_all(self)
        return self.search_results